"""
H-MAD DST Framework — Central Configuration
Scenario 1: Adversarial Gaslighting / Dialogue State Tracking
"""

from dataclasses import dataclass, field
from typing import Literal

# ─── Model Definitions ────────────────────────────────────────────────────────

MODELS = {
    "llama": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "ollama_tag": "llama3.1:8b",
        "hf_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "label": "Llama 3.1 8B",
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "ollama_tag": "qwen2.5:7b",
        "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen 2.5 7B",
    },
    "gemma": {
        "name": "google/gemma-2-2b-it",
        "ollama_tag": "gemma2:2b",
        "hf_model_id": "google/gemma-2-2b-it",
        "label": "Gemma 2 2B",
    },
}

# ─── Backend Configuration ─────────────────────────────────────────────────────

BACKEND: Literal["ollama", "huggingface"] = "huggingface"

OLLAMA_BASE_URL = "http://localhost:11434"

# Set via environment variable HF_API_TOKEN, or fill in here (not recommended in VCS)
HF_API_TOKEN = ""  # or os.environ.get("HF_API_TOKEN", "")
HF_BASE_URL = "https://router.huggingface.co"

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
    # {"proponent": "llama", "gaslighter": "qwen",  "judge": "qwen",  "config_id": "A"},
    {"proponent": "qwen",  "gaslighter": "llama", "judge": "gemma", "config_id": "B"},
    {"proponent": "llama", "gaslighter": "qwen",  "judge": "gemma", "config_id": "C"},
]

# Number of turns in each debate (Gaslighter gets a turn every 2 dialogue turns)
MAX_TURNS = 10

# At which turn does the Gaslighter introduce its first contradiction?
# (0-indexed, so turn 3 = 4th message)
GASLIGHTING_START_TURN = 3

# ─── Output Paths ──────────────────────────────────────────────────────────────

TRANSCRIPTS_DIR = "transcripts"
METRICS_DIR = "metrics"
REPORT_PATH = "report/report.html"
