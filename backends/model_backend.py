"""
backends/model_backend.py
Unified inference backend: Ollama (local) or HuggingFace Inference API.
"""

import os
import json
import time
import requests
from typing import Optional
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    BACKEND, MODELS,
    OLLAMA_BASE_URL, HF_BASE_URL, HF_API_TOKEN,
    GENERATION_CONFIG,
)


class ModelBackend:
    """
    Wraps either Ollama or HuggingFace Inference API.
    Usage:
        backend = ModelBackend(model_key="llama")
        response = backend.generate(messages=[...])
    """

    def __init__(self, model_key: str, backend: Optional[str] = None):
        """
        Args:
            model_key: One of "llama" or "qwen" (keys in config.MODELS).
            backend: "ollama" or "huggingface". Defaults to config.BACKEND.
        """
        self.model_key = model_key
        self.backend = backend or BACKEND
        self.model_info = MODELS[model_key]
        self._validate()

    def _validate(self):
        if self.backend not in ("ollama", "huggingface"):
            raise ValueError(f"Unknown backend: {self.backend}")
        if self.backend == "huggingface":
            token = HF_API_TOKEN or os.environ.get("HF_API_TOKEN", "")
            if not token:
                raise EnvironmentError(
                    "HF_API_TOKEN is not set. "
                    "Set it in config.py or export HF_API_TOKEN=<your_token>"
                )

    # ─── Public API ───────────────────────────────────────────────────────────

    def generate(self, messages: list[dict], system_prompt: Optional[str] = None) -> str:
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
            return self._generate_ollama(full_messages)
        else:
            return self._generate_hf(full_messages)

    # ─── Ollama ───────────────────────────────────────────────────────────────

    def _generate_ollama(self, messages: list[dict]) -> str:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": self.model_info["ollama_tag"],
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": GENERATION_CONFIG["temperature"],
                "top_p": GENERATION_CONFIG["top_p"],
                "num_predict": GENERATION_CONFIG["max_new_tokens"],
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

    # ─── HuggingFace Inference API ────────────────────────────────────────────

    def _generate_hf(self, messages: list[dict]) -> str:
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
            "max_tokens": GENERATION_CONFIG["max_new_tokens"],
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
