# Technical Approach & Limitations

## Overview

This pipeline implements a Retrieval-Augmented Generation (RAG) system for extracting structured ESG data from four steel company sustainability reports. The design prioritises auditability — every answer carries a source company, filename, and PDF page number — and numerical accuracy, which is substantially harder than text retrieval because numbers are context-sensitive: the same digit sequence means different things depending on which table row, fiscal year convention, and unit system surrounds it.

---

## PDF Loading

**Library choice: pdfplumber**

pdfplumber was chosen over pypdf and pdfminer.six for two reasons. First, it provides spatial awareness: it knows where on the page each word sits, which is essential for table detection. Second, it exposes `page.extract_tables()` which returns structured row/column data, not just a flat text stream. pdfminer.six offers similar capabilities but at a lower level of abstraction; pdfplumber wraps it ergonomically. PyMuPDF (fitz) would also work but carries an AGPL licence that may conflict with commercial deployment.

**Limitation:** `extract_text()` returns an empty string on scanned or image-only PDFs. Production fix: OCR via pytesseract after rasterising with pdftoppm. Not applied here as all four reports have a text layer.

---

## Chunking Strategy

**Table-aware chunking**

The core insight is that sustainability emissions data lives in tables, and a sliding-window chunker that splits on character count will cut table rows in half. A number stranded mid-row is unrecoverable — there is no semantic overlap that would stitch it back together. The fix is to extract tables first as atomic units, then apply the sliding window only to the remaining prose.

```
For each page:
  1. page.extract_tables()  →  each table → one chunk (never split)
  2. remaining text         →  sliding window (512 chars, 64 overlap)
```

**Why within-page only**

Chunks never cross page boundaries. This guarantees that every chunk has a single, unambiguous page number as its citation. This is a deliberate constraint: ESG data is auditable information, and "approximately page 140" is not a valid citation.

**Why 512 characters**

At roughly 80–120 words per chunk, 512 characters captures one to two table rows with their surrounding context, or a short paragraph of narrative text. Shorter chunks improve retrieval precision but risk splitting a multi-row table entry. Longer chunks reduce precision by including more off-topic text alongside the target number.

**Tradeoff not taken: sentence or paragraph chunking**

Sentence-based chunking breaks on PDF extraction artifacts (mid-sentence line breaks, hyphenation, navigation headers mixed into body text). Paragraph-based chunking is unreliable because pdfplumber does not preserve paragraph structure — it returns text flow, not document structure. Semantic chunking (embedding-based splits) would require encoding the document twice and was not worth the cost given the corpus size.

**Production approach:** structure-aware coarse splits on section headings, then fixed-size within sections — the pattern used by LlamaIndex and similar frameworks.

---

## Embedding & Vector Store

**Model: all-MiniLM-L6-v2**

A lightweight sentence-transformers model (22M parameters, 384-dim embeddings). Fast enough to embed ~4,000 chunks in under two minutes on CPU. Chosen for fair model comparison: all LLMs are evaluated against the same retrieval backbone, isolating generation quality as the variable.

**Limitation:** all-MiniLM-L6-v2 is English-optimised. POSCO's report contains Korean-language pages, producing weaker embeddings for those sections. Production fix: paraphrase-multilingual-MiniLM-L12-v2.

**Index: FAISS IndexFlatIP (exact cosine)**

The corpus contains ~4,000 vectors at 384 dimensions — approximately 6 MB. At this scale, exact nearest-neighbour search is fast and correct. Approximate indices (IndexIVFFlat, HNSW) are only justified above ~500k vectors where exact search becomes a bottleneck.

**Stratified retrieval**

A standard FAISS query against the full index would over-represent JFE Holdings, whose 311-page report produces ~10× more chunks than JSW Steel (18 pages). To guarantee representation from all four companies, four per-company sub-indices are built at index time (sliced from the same embedding matrix, no re-encoding). Each query retrieves top-k from each sub-index independently, then merges results.

---

## Hybrid Retrieval

**Problem with pure semantic search**

FAISS retrieves chunks that are semantically similar to the query. "Scope 1 emissions 2024" may retrieve chunks discussing Scope 1 methodology without containing the actual numbers. Numerical data has low semantic entropy — the numbers themselves carry almost no meaning to an embedding model.

**BM25 addition**

BM25 (Okapi BM25, via rank-bm25) performs keyword overlap retrieval. It scores on exact token matches, which is precisely what numerical extraction needs: "53,100,751" or "FY2024-25" will rank highly against a query containing those strings, regardless of surrounding semantic context. FAISS and BM25 retrieve independently (top-k each), then deduplicate before reranking.

**Cross-encoder reranking**

FAISS and BM25 each embed query and chunks separately. A cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) evaluates each (query, chunk) pair jointly — the query tokens attend to chunk tokens — producing more accurate relevance scores. The pipeline retrieves up to 80 candidates (10 FAISS + 10 BM25 × 4 companies), then the reranker trims to 40 for the LLM context. This also constrains context length, reducing hallucination pressure on the LLM.

**Overhead:** BM25 + cross-encoder add ~15 seconds per query. Acceptable since GPU time is reserved for LLM generation.

---

## Q1 Sub-queries

Q1 asks for Scope 1, 2, 3 emissions and energy for 4 companies × 3 years = 48 data points in a single query. With 40 context chunks covering all four companies simultaneously, the LLM must parse four different table formats, four different fiscal year conventions, and four different unit systems at once. Observed failure modes: wrong table row selection, year-label misreading, unit confusion.

The fix is to decompose Q1 into four per-company sub-queries. Each sub-query's 40 context slots are focused entirely on one company. A minimal reporting hint (fiscal year alignment and unit system) is prepended to each sub-query system prompt — these are metadata facts about the report, not extracted values.

Q2 and Q3 are run as single queries; they perform well with the full 40-chunk context.

---

## LLM Inference

**Framework: vLLM**

vLLM provides efficient PagedAttention-based inference for transformer models. All models are loaded with `local_files_only=True` — no network calls at inference time.

**Sequential loading**

Models are loaded one at a time and explicitly unloaded (del + torch.cuda.empty_cache()) between runs. A single H100 80GB cannot hold two large models simultaneously.

**Quantization**

Models exceeding ~60 GB in fp16/bfloat16 are loaded in 4-bit bitsandbytes quantization. This reduces memory by ~4× at the cost of some numerical precision. Affected models: llama_70b (70B), qwen_32b (32B), mistral_large (123B).

**Prompt design**

The system prompt explicitly instructs the model that the numbered excerpts are source material to synthesise, not items to respond to individually. Without this instruction, smaller models (particularly Mistral 7B) produce one answer per numbered chunk rather than one unified answer — a behaviour caused by the numbered list format mimicking instruction-following templates from training data.

**Temperature: 0.1**

Near-deterministic output. High temperature introduces variability in number extraction, which is undesirable for auditable ESG data.

---

## Agent Routing (Bonus)

A rule-based router (`route_query` in the notebook) classifies incoming queries into four actions without calling an LLM:

- `retrieve_from_pdfs` — query mentions a known company name and an ESG topic
- `web_search` — query references live data, stock prices, or post-publication events
- `structured_extraction` — query requests tabular or JSON output
- `request_clarification` — query is ambiguous (no specific company or time frame)

This is deterministic and zero-cost. A production system would use an LLM tool-selection call for robustness (handling abbreviations, synonyms, implicit references), but the rule-based approach is sufficient and transparent for the three fixed assignment questions. The most robust alternative is query expansion: rewrite the question into four company-specific sub-queries and run all, eliminating the need for routing entirely — the pattern used by HyDE and LlamaIndex MultiStepQueryEngine.

---

## Observed Failure Modes

**JFE table structure**

JFE's Scope 1+2 table (PDF p.259) has multiple row groups: ST Gr (JFE Steel Group), ST (JFE Steel standalone), and subsidiaries. All models retrieved this page but consistently selected the wrong row — either the combined total or the standalone figure — rather than the "All group" row. This is a table parsing problem, not a retrieval problem: the correct chunk is retrieved, but the LLM cannot reliably navigate a multi-row-group table from TSV text alone.

**JSW Indian number notation**

JSW reports emissions in Indian lakh-crore notation: 5,31,00,751 = 53,100,751 (not 5.3 million). The comma placement follows lakh grouping (2-2-3 from right) rather than Western thousands grouping (3-3-3). Models that do not recognise this notation produce values off by one or two orders of magnitude. The fiscal year hint in Q1 sub-queries flags this notation; it is not fully resolved.

**Tata year-shift**

Tata's data table (PDF p.52) shows five columns: FY21, FY22, FY23, FY24, FY25. The table header is not always parsed into the chunk. Models without the fiscal year hint assign values to the wrong calendar year, producing a systematic off-by-one error across all fields.

**POSCO Scope 1+2 separation**

POSCO presents direct (Scope 1) and indirect (Scope 2) emissions in adjacent rows of the same table. Several models combined them into a single Scope 1+2 total rather than extracting them separately. The retrieval correctly returns p.139, but the TSV table representation does not make the row semantics explicit enough for the model to distinguish.

**JSW/POSCO net-zero targets (Q3)**

Both JSW Steel ("Net Neutral by 2050") and POSCO ("Carbon Neutral by 2050") were missed by all models. The interim 2030 targets appear on the same pages as the emissions data tables (which are heavily retrieved for Q1), while the 2050 commitment language appears on different pages that rank lower for the Q3 query vector. This is a retrieval failure: the correct information exists in the corpus but is not surfaced. Fix: boost pages containing "2050", "net neutral", or "carbon neutral" in the Q3 BM25 query.

**POSCO Korean-language pages**

POSCO's report contains Korean-language sections. The all-MiniLM-L6-v2 embedding model is English-optimised, producing weaker similarity scores for Korean chunks even when those chunks contain the relevant data. This reduces recall for POSCO across all questions.

---

## Model Performance Summary

Seven models were evaluated: Mistral 7B, Llama 8B, Qwen 14B, Gemma 27B, Llama 70B, Qwen 32B, and Mistral Large 123B.

**Q1 (numerical extraction):** Larger model size does not consistently improve Q1 accuracy. Gemma 27B achieves the best JSW extraction (all scope values correct in lakh notation). Mistral Large 123B performs worst on Q1, incorrectly deriving Scope 1 by subtracting Scope 2 from a misidentified total. The primary driver of Q1 accuracy is table chunk quality, not model capacity.

**Q2 (CCUS identification):** All models correctly identify all four companies as having CCUS activity. Q2 is effectively solved at all model sizes.

**Q3 (net-zero targets):** Larger models (Llama 70B, Mistral Large 123B) correctly identify JFE Holdings' 2050 target, which smaller models miss. All models correctly identify Tata Steel UK's 2045 target. JSW and POSCO 2050 targets are missed by all models due to retrieval failure.

---

## Scaling to Production

```python
# Persist FAISS index — avoids ~2 min rebuild on each run
import faiss, pickle
faiss.write_index(vs.index, "esg_index.faiss")
with open("esg_chunks.pkl", "wb") as f:
    pickle.dump(vs.chunks, f)

# Reload
vs.index = faiss.read_index("esg_index.faiss")
with open("esg_chunks.pkl", "rb") as f:
    vs.chunks = pickle.load(f)
```

Additional production considerations: multilingual embedding model for Korean/Japanese content; OCR fallback for scanned pages; structured output enforcement (JSON mode or grammar-constrained decoding) to guarantee parseable table output; page-neighbourhood retrieval to ensure adjacent table pages (e.g., JFE pp.259–263) are always retrieved together.
