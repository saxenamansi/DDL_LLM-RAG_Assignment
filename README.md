# ESG RAG Pipeline — LLM/NLP Developer Technical Assignment

Retrieval-Augmented Generation pipeline for extracting sustainability and ESG data from four steel company reports.

**Companies:** JFE Holdings · JSW Steel · POSCO · Tata Steel UK

---

## Repository Structure

```
8_DDL-EnviroLab/
├── Config.py              # All configuration: models, paths, chunking parameters
├── DataClasses.py         # Shared dataclasses (PageRecord, Chunk)
├── Chunking.py            # PDF loading + table-aware chunking
├── VectorStore.py         # FAISS + BM25 hybrid retrieval with cross-encoder reranking
├── RAGEngine.py           # Prompt construction, LLM inference, multi-model runner
├── GroundTruth.py         # Pre-extracted reference answers for evaluation
├── Evaluator.py           # Automated accuracy scoring against ground truth
├── RAG_updated.ipynb      # Main notebook — run this end to end
├── requirements.txt       # Python dependencies
├── outputs/               # Per-model answer .txt files, CSV/JSON exports
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.12
- NVIDIA GPU with ≥80GB VRAM (tested on H100 80GB SXM5)
- CUDA 12.x
- HuggingFace model weights downloaded locally (pipeline uses `local_files_only=True`)

### Environment

The pipeline was developed and tested on an HPC cluster running SLURM. The environment is a Python venv layered on top of a conda base.

```bash
# Create and activate the environment
python -m venv ddl_venv
source ddl_venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Register the kernel with Jupyter
python -m ipykernel install --user --name=ddl_venv
```

### PDF reports

Place the four PDF files in the project root directory:

```
JFE Holdings_Sustainability_2025.pdf
JSW Steel_Sustainability_2025.pdf
POSCO_Sustainability_2024.pdf
Tata Steel_Sustainability_2025.pdf
```

The PDFs are not included in this repository (copyright). Obtain them from the company sustainability portals or the drive provided with this assignment.

---

## Running the Pipeline

Open `RAG_updated.ipynb` and run cells top to bottom. Each section is self-contained:

| Section | What it does |
|---|---|
| 1. Install Dependencies | One-time installs (commented out after first run) |
| 2. Imports | Loads all modules |
| 3. PDF Loading & Chunking | Extracts tables + text from all four PDFs |
| 4. Chunking | Saves chunk files to `chunks/` for inspection |
| 5. Embedding & FAISS | Builds vector store + BM25 index + loads reranker |
| 6–7. Questions & Run | Runs all 7 models sequentially (~2–4 hours total) |
| 8–9. Outputs | Prints answers, exports CSV/JSON |
| 10. Agent Routing | Demo of query routing logic |
| 12. Ground Truth | Loads verified reference answers |
| 13. Approach & Limitations | Discussion notes |

**Expected runtime:** approximately 10–20 minutes per small model (7–27B), 40–90 minutes per large quantized model (70B+). Total wall time ~4 hours for all 7 models on a single H100.

---

## Models

All models are run via [vLLM](https://github.com/vllm-project/vllm) and loaded sequentially — one at a time — to fit on a single GPU.

| Key | Model | Size | Precision |
|---|---|---|---|
| mistral_7b | mistralai/Mistral-7B-Instruct-v0.3 | 7B | float16 |
| llama_8b | meta-llama/Llama-3.1-8B-Instruct | 8B | float16 |
| qwen_14b | Qwen/Qwen3-14B | 14B | float16 |
| gemma_27b | google/gemma-3-27b-it | 27B | bfloat16 |
| llama_70b | meta-llama/Llama-3.1-70B-Instruct | 70B | 4-bit (bitsandbytes) |
| qwen_32b | Qwen/Qwen3-32B | 32B | 4-bit (bitsandbytes) |
| mistral_large | mistralai/Mistral-Large-Instruct-2407 | 123B | 4-bit (bitsandbytes) |

Models must be downloaded from HuggingFace and available locally. The pipeline does not download at runtime.

---

## Outputs

After running Section 7, the following files are written to `outputs/`:

- `{model_key}_answers.txt` — per-model plain-text answers with page references
- `model_comparison_results.csv/.json` — all answers in tabular form
- `model_comparison_pivot.csv` — side-by-side comparison across models
- `model_latency.csv` — per-model, per-question latency
- `evaluation_scores.csv` — automated accuracy scores against ground truth

---

## No API Keys

This pipeline runs entirely locally. No external API calls are made. Do not add API keys to any file in this repository.
