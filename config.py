"""
H-MAD DST Framework — Central Configuration
Scenario 1: Adversarial Gaslighting / Dialogue State Tracking
"""

import os
import threading
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# ─── Model Definitions ────────────────────────────────────────────────────────

MODELS = {
    "llama": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "ollama_tag": "llama3.1:8b",
        "hf_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "label": "Llama 3.1 8B (Q4)",
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "ollama_tag": "qwen2.5:7b",
        "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen 2.5 7B (Q4)",
    },
    "gpt-oss": {
        "name": "openai/gpt-oss-20b",
        "groq_model_id": "openai/gpt-oss-20b",
        "label": "GPT-oss 20B",
    },
}

# ─── Backend Configuration ─────────────────────────────────────────────────────

# Default backend for debate agents (Proponent & Gaslighter)
BACKEND: Literal["ollama", "huggingface"] = "ollama"

OLLAMA_BASE_URL = "http://localhost:11434"

# HuggingFace (kept for backward compatibility)
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_BASE_URL = "https://router.huggingface.co"

# Groq (used for the Judge — GPT-oss-20B)
# Supports multiple comma-separated keys for rate-limit rotation
_raw_groq_keys = os.environ.get("GROQ_API_KEY", "")
GROQ_API_KEYS = [k.strip() for k in _raw_groq_keys.split(",") if k.strip()]
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""   # backward compat
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqKeyRotator:
    """Thread-safe round-robin rotator for multiple Groq API keys."""

    def __init__(self, keys: list[str]):
        self._keys = keys
        self._index = 0
        self._lock = threading.Lock()

    @property
    def current_key(self) -> str:
        if not self._keys:
            raise EnvironmentError("No GROQ_API_KEYs configured.")
        with self._lock:
            return self._keys[self._index % len(self._keys)]

    def rotate(self) -> str:
        """Advance to the next key and return it."""
        if len(self._keys) <= 1:
            return self.current_key
        with self._lock:
            self._index = (self._index + 1) % len(self._keys)
            new_key = self._keys[self._index]
        print(f"   [GroqKeyRotator] Rotated to key …{new_key[-6:]}")
        return new_key

    def __len__(self):
        return len(self._keys)


groq_key_rotator = GroqKeyRotator(GROQ_API_KEYS)

# Inference params
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

# ─── Experiment Configuration ──────────────────────────────────────────────────

# Both role configurations will be run
CONFIGURATIONS = [
    {"proponent": "llama", "gaslighter": "qwen",  "judge": "gpt-oss", "config_id": "A"},
    {"proponent": "qwen",  "gaslighter": "llama", "judge": "gpt-oss", "config_id": "B"},
]

# Number of turns in each debate (Gaslighter gets a turn every 2 dialogue turns)
MAX_TURNS = 10

# At which turns can the Gaslighter introduce its first contradiction?
# (0-indexed range, e.g., (2, 5) means it can start between turn 2 and 5)
GASLIGHTING_RANGE = (2, 5)

# ─── Output Paths ──────────────────────────────────────────────────────────────

TRANSCRIPTS_DIR = "transcripts"
METRICS_DIR = "metrics"
REPORT_PATH = "report/report.html"
