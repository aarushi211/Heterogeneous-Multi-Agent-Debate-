"""
backends/model_backend.py
Unified inference backend: Ollama, HuggingFace Inference API, or Groq.
"""

import os
import json
import re
import time
import requests
from typing import Optional
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    BACKEND, MODELS,
    OLLAMA_BASE_URL, HF_BASE_URL, HF_API_TOKEN,
    GROQ_API_KEY, GROQ_BASE_URL, groq_key_rotator,
    GENERATION_CONFIG,
)


class ModelBackend:
    """
    Wraps Ollama, HuggingFace Inference API, or Groq API.
    Usage:
        backend = ModelBackend(model_key="llama")
        response = backend.generate(messages=[...])
    """

    def __init__(self, model_key: str, backend: Optional[str] = None):
        """
        Args:
            model_key: One of "llama", "qwen", or "gpt-oss" (keys in config.MODELS).
            backend: "ollama", "huggingface", or "groq". Defaults to config.BACKEND.
        """
        self.model_key = model_key
        self.backend = backend or BACKEND
        self.model_info = MODELS[model_key]
        self._validate()

    def _validate(self):
        if self.backend not in ("ollama", "huggingface", "groq"):
            raise ValueError(f"Unknown backend: {self.backend}")
        if self.backend == "huggingface":
            token = HF_API_TOKEN or os.environ.get("HF_API_TOKEN", "")
            if not token:
                raise EnvironmentError(
                    "HF_API_TOKEN is not set. "
                    "Set it in .env or export HF_API_TOKEN=<your_token>"
                )
        if self.backend == "groq":
            if len(groq_key_rotator) == 0:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. "
                    "Set it in .env or export GROQ_API_KEY=<your_key>"
                )
            print(f"   [Groq] {len(groq_key_rotator)} API key(s) available for rotation.")

    # ─── Public API ───────────────────────────────────────────────────────────

    def generate(self, messages: list[dict], system_prompt: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """
        Generate a response from the model.

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."} dicts.
                      Do NOT include a system message here; pass it via system_prompt.
            system_prompt: Optional system-level instruction.

        Returns:
            The model's response as a plain string.
        """
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = messages

        if self.backend == "ollama":
            return self._generate_ollama(full_messages, max_tokens)
        elif self.backend == "groq":
            return self._generate_groq(full_messages, max_tokens)
        else:
            return self._generate_hf(full_messages, max_tokens)

    # ─── Ollama ───────────────────────────────────────────────────────────────

    def _generate_ollama(self, messages: list[dict], max_tokens: Optional[int] = None) -> str:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": self.model_info["ollama_tag"],
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": GENERATION_CONFIG["temperature"],
                "top_p": GENERATION_CONFIG["top_p"],
                "num_predict": max_tokens or GENERATION_CONFIG["max_new_tokens"],
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
                "Make sure `ollama serve` is running."
            )
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    # ─── Groq API ─────────────────────────────────────────────────────────────

    def _generate_groq(self, messages: list[dict], max_tokens: Optional[int] = None) -> str:
        model_id = self.model_info.get("groq_model_id", self.model_info["name"])
        url = f"{GROQ_BASE_URL}/chat/completions"

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens or GENERATION_CONFIG["max_new_tokens"],
            "temperature": GENERATION_CONFIG["temperature"],
            "top_p": GENERATION_CONFIG["top_p"],
        }

        # Total attempts = retries_per_key × number_of_keys (at least 3)
        num_keys = max(len(groq_key_rotator), 1)
        retries_per_key = 2
        max_retries = max(num_keys * retries_per_key, 3)

        for attempt in range(max_retries):
            key = groq_key_rotator.current_key
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)

                # 429 (Too Many Requests) or 423 (Locked / rate-limited)
                if resp.status_code in (429, 423):
                    groq_key_rotator.rotate()
                    wait = 5 * (attempt + 1)
                    print(
                        f"   [Groq] Rate limited ({resp.status_code}) on attempt "
                        f"{attempt+1}/{max_retries}, switching key & retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code == 503:
                    wait = 15 * (attempt + 1)
                    print(f"   [Groq] Service unavailable, retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"   [Groq] Timeout on attempt {attempt+1}, retrying...")
                    time.sleep(5)
                else:
                    raise TimeoutError(f"Groq API timed out after {max_retries} attempts.")
            except Exception as e:
                error_msg = resp.text if 'resp' in locals() else 'N/A'
                raise RuntimeError(f"Groq generation failed: {e}\nResponse: {error_msg}")

        raise RuntimeError("Groq generation failed after all retries (all keys exhausted).")

    # ─── HuggingFace Inference API ────────────────────────────────────────────

    def _generate_hf(self, messages: list[dict], max_tokens: Optional[int] = None) -> str:
        token = HF_API_TOKEN or os.environ.get("HF_API_TOKEN", "")
        model_id = self.model_info["hf_model_id"]

        # FIX: The router uses a single unified endpoint for all models
        url = "https://router.huggingface.co/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        # The model_id must be specified here for the router to direct the request
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens or GENERATION_CONFIG["max_new_tokens"],
            "temperature": GENERATION_CONFIG["temperature"],
            "top_p": GENERATION_CONFIG["top_p"],
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Note: The timeout remains 120s which is good for larger models
                resp = requests.post(url, headers=headers, json=payload, timeout=300)

                if resp.status_code == 503:
                    wait = 20 * (attempt + 1)
                    print(f"   [HF] Model loading, retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                return data["choices"][0]["message"]["content"].strip()

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"   [HF] Timeout on attempt {attempt+1}, retrying...")
                    time.sleep(5)
                else:
                    raise TimeoutError(f"HF API timed out after {max_retries} attempts.")
            except Exception as e:
                # Improved error reporting to see exactly what the Router is complaining about
                error_msg = resp.text if 'resp' in locals() else 'N/A'
                raise RuntimeError(f"HuggingFace generation failed: {e}\nResponse: {error_msg}")

        raise RuntimeError("HuggingFace generation failed after all retries.")

    def __repr__(self):
        return f"<ModelBackend model={self.model_info['label']} backend={self.backend}>"

