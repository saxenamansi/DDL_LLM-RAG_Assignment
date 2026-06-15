"""
Config.py
=========
All configuration constants for the ESG RAG pipeline.
Edit this file to change models, paths, or chunking parameters.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

PDF_DIR   = Path(".")
CHUNK_DIR = Path("chunks")
OUTPUT_DIR = Path("outputs")

# ── PDF corpus ─────────────────────────────────────────────────────────────────

PDF_FILES = {
    "JFE Holdings":  "JFE Holdings_Sustainability_2025.pdf",
    "JSW Steel":     "JSW Steel_Sustainability_2025.pdf",
    "POSCO":         "POSCO_Sustainability_2024.pdf",
    "Tata Steel UK": "Tata Steel_Sustainability_2025.pdf",
}

# ── Chunking ───────────────────────────────────────────────────────────────────

CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 64

# ── Retrieval ──────────────────────────────────────────────────────────────────

TOP_K_PER_COMPANY = 10    # 10 chunks × 4 companies = 40 chunks per query

# ── Embedding model ────────────────────────────────────────────────────────────

EMBED_MODEL = "all-MiniLM-L6-v2"

# ── LLM registry ───────────────────────────────────────────────────────────────
# All models run sequentially — one loaded at a time to fit on a single GPU.
# Models that exceed ~60GB in fp16 are loaded with 4-bit quantization (bitsandbytes).
# This is noted in the output metadata so results can be interpreted accordingly.

MODEL_MAP = {
    "mistral_7b":     "mistralai/Mistral-7B-Instruct-v0.3",
    "llama_8b":       "meta-llama/Llama-3.1-8B-Instruct",
    "qwen_14b":       "Qwen/Qwen3-14B",
    "gemma_27b":      "google/gemma-3-27b-it",
    "llama_70b":      "meta-llama/Llama-3.1-70B-Instruct",
    "qwen_32b":       "Qwen/Qwen3-32B",
    "mistral_large":  "mistralai/Mistral-Large-Instruct-2407",
}

# Approximate parameter counts for metadata logging
MODEL_SIZE_B = {
    "mistral_7b":    "7B",
    "llama_8b":      "8B",
    "qwen_14b":      "14B",
    "gemma_27b":     "27B",
    "llama_70b":     "70B",
    "qwen_32b":      "32B",
    "mistral_large": "123B",
}

# fp16 VRAM requirements exceed H100 80GB for these — load in 4-bit instead
QUANTIZED_MODELS = {"llama_70b", "qwen_32b", "mistral_large"}

# dtype per model (before quantization)
MODEL_DTYPE = {
    "mistral_7b":    "float16",
    "llama_8b":      "float16",
    "qwen_14b":      "float16",
    "gemma_27b":     "bfloat16",
    "llama_70b":     "bfloat16",
    "qwen_32b":      "bfloat16",
    "mistral_large": "bfloat16",
}

# vLLM max_model_len overrides — set only where needed to avoid OOM
MODEL_MAX_LEN = {
    "gemma_27b": 62544,   # default context exceeds H100 VRAM; cap it
}

# ── Sampling ───────────────────────────────────────────────────────────────────

SAMPLING_CONFIG = {
    "temperature":        0.1,
    "top_p":              0.9,
    "max_tokens":         4096,
    "repetition_penalty": 1.1,
}