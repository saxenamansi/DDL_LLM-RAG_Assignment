"""
Config.py
=========
All configuration constants for the ESG RAG pipeline.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

PDF_DIR    = Path(".")
CHUNK_DIR  = Path("chunks")
OUTPUT_DIR = Path("outputs")

# ── PDF corpus ─────────────────────────────────────────────────────────────────

PDF_FILES = {
    "JFE Holdings":  "JFE Holdings_Sustainability_2025.pdf",
    "JSW Steel":     "JSW Steel_Sustainability_2025.pdf",
    "POSCO":         "POSCO_Sustainability_2024.pdf",
    "Tata Steel UK": "Tata Steel_Sustainability_2025.pdf",
}

# ── Chunking ───────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

# ── Retrieval ──────────────────────────────────────────────────────────────────

TOP_K_PER_COMPANY = 10   # per company; 10 × 4 companies = 40 chunks per query

# ── Embedding model ────────────────────────────────────────────────────────────

EMBED_MODEL = "all-MiniLM-L6-v2"

# ── LLM registry ───────────────────────────────────────────────────────────────
# Models run sequentially — one at a time to fit on a single H100 80GB.
# Models in QUANTIZED_MODELS are loaded in 4-bit (bitsandbytes) to fit in VRAM.

MODEL_MAP = {
    # "mistral_7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    # "llama_8b":     "meta-llama/Llama-3.1-8B-Instruct",
    # "qwen_14b":     "Qwen/Qwen3-14B",
    "gemma_27b":      "google/gemma-3-27b-it",
    "llama_70b":      "meta-llama/Llama-3.1-70B-Instruct",
    "qwen_32b":       "Qwen/Qwen3-32B",
    "mistral_large":  "mistralai/Mistral-Large-Instruct-2407",
}

MODEL_SIZE_B = {
    "mistral_7b":    "7B",
    "llama_8b":      "8B",
    "qwen_14b":      "14B",
    "gemma_27b":     "27B",
    "llama_70b":     "70B",
    "qwen_32b":      "32B",
    "mistral_large": "123B",
}

QUANTIZED_MODELS = {"llama_70b", "qwen_32b", "mistral_large"}

MODEL_DTYPE = {
    "mistral_7b":    "float16",
    "llama_8b":      "float16",
    "qwen_14b":      "float16",
    "gemma_27b":     "bfloat16",
    "llama_70b":     "bfloat16",
    "qwen_32b":      "bfloat16",
    "mistral_large": "bfloat16",
}

# vLLM max_model_len overrides to avoid OOM at load time
MODEL_MAX_LEN = {
    "gemma_27b":     62544,
    "llama_70b":      8192,
    "mistral_large":  8192,
    "qwen_32b":      16384,
}

# ── Sampling ───────────────────────────────────────────────────────────────────

SAMPLING_CONFIG = {
    "temperature":        0.1,
    "top_p":              0.9,
    "max_tokens":         4096,
    "repetition_penalty": 1.1,
}
