# ESG RAG Pipeline

Retrieval-Augmented Generation pipeline for extracting sustainability data from four steel company reports.

**Companies:** JFE Holdings · JSW Steel · POSCO · Tata Steel UK

---

## Repository Structure

```
DDL_LLM-RAG_Assignment/
├── Config.py              # Models, paths, chunking parameters
├── DataClasses.py         # Shared dataclasses (PageRecord, Chunk)
├── Chunking.py            # PDF loading + table-aware chunking
├── VectorStore.py         # FAISS + BM25 hybrid retrieval with cross-encoder reranking
├── RAGEngine.py           # Prompt construction, LLM inference, multi-model runner
├── GroundTruth.py         # Verified reference answers for all three questions
├── LLM-RAG.ipynb          # Main notebook — run end to end
├── Discussion.pdf         # Technical design, implementation choices & limitations
├── ANSWERS.md             # Final answers to all three questions (two best models)
├── requirements.txt       # Python dependencies
└── outputs/               # Per-model answer .txt files, CSV/JSON exports
```

---

## Setup

**Requirements:** Python 3.12, NVIDIA GPU ≥80GB VRAM (tested on H100 80GB), CUDA 12.x.

```bash
python -m venv ddl_venv
source ddl_venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=ddl_venv
```

### Download models

Models must be available locally before running. The pipeline will not download at runtime.

```python
from huggingface_hub import snapshot_download

models = [
    "google/gemma-3-27b-it",                    # requires Google licence acceptance
    "meta-llama/Llama-3.1-70B-Instruct",        # requires Meta licence acceptance
    "Qwen/Qwen3-32B",
    "mistralai/Mistral-Large-Instruct-2407",
]
for model_id in models:
    snapshot_download(model_id)   # downloads to ~/.cache/huggingface/hub/
```

Log in first if models are gated:

```bash
huggingface-cli login
```

### PDF reports

Place the four PDFs in the project root:

```
JFE Holdings_Sustainability_2025.pdf
JSW Steel_Sustainability_2025.pdf
POSCO_Sustainability_2024.pdf
Tata Steel_Sustainability_2025.pdf
```

The PDFs are not included in this repository. They are provided separately with this assignment.

---

## Running the Pipeline

```bash
# 1. Navigate to the project directory
cd DDL_LLM-RAG_Assignment

# 2. Activate the environment
source ddl_venv/bin/activate

# 3. Launch Jupyter
jupyter lab
# or: jupyter notebook
```

Open `LLM-RAG.ipynb` and run all cells top to bottom (**Kernel → Restart & Run All**, or run each cell in order).

Make sure the four PDF files and all model weights are in place before running — the pipeline will error immediately if either is missing.

| Section | What it does |
|---|---|
| 1. Install Dependencies | One-time installs (commented out after first run) |
| 2. Imports | Loads all modules |
| 3. PDF Loading & Chunking | Extracts tables + text from all PDFs |
| 4. Chunking | Saves chunks to `chunks/` for inspection |
| 5. Embedding & Vector Index | Builds FAISS + BM25 + loads reranker |
| 6–7. Questions & Run | Runs all models sequentially |
| 8–9. Outputs | Prints answers, exports CSV/JSON |
| 10. Agent Routing | Query routing demo |
| 11. Ground Truth | Reference answers for all three questions |
| 12. Approach & Limitations | Discussion notes |

Full verbose output (model loading, per-question answers, errors) is written to `rag_run.log` in the project root. The notebook prints a short progress summary only.

**Expected runtime:** ~10–20 min per small model, ~40–90 min per large quantised model. Total wall time for all four large models on a single H100 is approximately 3–4 hours.

---

## Models

| Key | Model | Size | Precision |
|---|---|---|---|
| gemma_27b | google/gemma-3-27b-it | 27B | bfloat16 |
| llama_70b | meta-llama/Llama-3.1-70B-Instruct | 70B | 4-bit |
| qwen_32b | Qwen/Qwen3-32B | 32B | 4-bit |
| mistral_large | mistralai/Mistral-Large-Instruct-2407 | 123B | 4-bit |

Small models (7B–14B) are included in `Config.py` but commented out. Uncomment to include them.

---

## Outputs

Written to `outputs/` after Section 7:

- `{model_key}_answers.txt` — plain-text answers with page references
- `model_comparison_results.csv / .json` — all answers in tabular form
- `model_comparison_pivot.csv` — side-by-side answer comparison
- `model_latency.csv` — per-model, per-question latency

---

## Documentation

- `Discussion.pdf` — full technical write-up covering design decisions, retrieval strategy, prompt engineering, observed failure modes, and model performance analysis
- `ANSWERS.md` — final answers to all three questions from the two best-performing models (Llama 3.1 70B and Mistral Large 123B), with page-level citations and known limitations

---

## No API Keys

This pipeline runs entirely locally. No external API calls are made.
