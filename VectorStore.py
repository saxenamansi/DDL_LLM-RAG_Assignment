"""
VectorStore.py
===============
FAISS-backed vector store with stratified per-company retrieval,
BM25 hybrid retrieval, and cross-encoder reranking.

Class
-----
VectorStore
    build()    : embed all chunks, build full index + one sub-index per company
                 + one BM25 index per company
    retrieve() : FAISS top-k + BM25 top-k per company → deduplicate → rerank
                 → return top_k_per_company × n_companies chunks
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from Config import EMBED_MODEL, TOP_K_PER_COMPANY
from DataClasses import Chunk


RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class VectorStore:
    """
    Hybrid retrieval: FAISS (semantic) + BM25 (keyword), reranked by a
    cross-encoder before passing to the LLM.

    Build strategy
    --------------
    - Full FAISS index kept for reference/debugging.
    - Per-company FAISS sub-index: sliced from the embedding matrix,
      no re-encoding needed.
    - Per-company BM25 index: built from lowercased whitespace-tokenised text.
      Stored alongside the chunk list it was built from so index positions
      map directly back to Chunk objects.

    Retrieval strategy
    ------------------
    For each company:
      1. FAISS top-k  (semantic similarity via normalised inner product)
      2. BM25  top-k  (keyword overlap — strong for exact numbers / fiscal years)
      3. Deduplicate by chunk text across both lists
    Then globally:
      4. Cross-encoder reranks all candidates jointly against the query
      5. Return top (top_k_per_company × n_companies) after reranking

    Default k=10 per retriever per company gives up to 80 candidates before
    reranking (10 FAISS + 10 BM25) × 4 companies, deduplicated down to ≤80,
    then reranked and trimmed to 40 for the prompt.
    """

    def __init__(
        self,
        model_name:   str  = EMBED_MODEL,
        rerank_model: str  = RERANK_MODEL,
        rerank:       bool = True,
    ):
        print(f"Loading embedding model: {model_name} …")
        self.model = SentenceTransformer(model_name)

        self.rerank = rerank
        if rerank:
            print(f"Loading reranker: {rerank_model} …")
            self.reranker = CrossEncoder(rerank_model)
        else:
            self.reranker = None

        # FAISS
        self.index:           Optional[faiss.IndexFlatIP]                      = None
        self.chunks:          List[Chunk]                                       = []
        self.company_indices: Dict[str, Tuple[faiss.IndexFlatIP, List[Chunk]]] = {}

        # BM25 — parallel structure: one index + one chunk list per company
        self.bm25_indices:    Dict[str, BM25Okapi]   = {}
        self.bm25_chunks:     Dict[str, List[Chunk]]  = {}

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: List[Chunk], batch_size: int = 256) -> None:
        """Embed all chunks; build FAISS full + per-company + BM25 per-company."""

        texts = [c.text for c in chunks]
        print(f"Embedding {len(texts)} chunks (batch_size={batch_size}) …")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine sim ≡ dot product on unit vectors
            convert_to_numpy=True,
        )
        dim = embeddings.shape[1]

        # ── Full FAISS index (reference / debugging) ───────────────────────
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks
        print(f"Full FAISS index built: {self.index.ntotal} vectors, dim={dim}")

        # ── Per-company FAISS sub-indices + BM25 indices ───────────────────
        buckets: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            buckets[c.company].append(i)

        for company, positions in buckets.items():
            company_chunks = [chunks[i] for i in positions]

            # FAISS sub-index
            sub_embs  = embeddings[positions].astype(np.float32)
            sub_index = faiss.IndexFlatIP(dim)
            sub_index.add(sub_embs)
            self.company_indices[company] = (sub_index, company_chunks)

            # BM25 index — tokenise once at build time
            tokenized = [c.text.lower().split() for c in company_chunks]
            self.bm25_indices[company] = BM25Okapi(tokenized)
            self.bm25_chunks[company]  = company_chunks

            print(
                f"  [{company}] FAISS: {sub_index.ntotal} vectors | "
                f"BM25: {len(tokenized)} docs"
            )

    # ── Retrieve ───────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:              str,
        top_k_per_company:  int = TOP_K_PER_COMPANY,
    ) -> List[Chunk]:
        """
        Hybrid retrieval with reranking.

        Steps
        -----
        1. Encode query once for FAISS.
        2. Per company: FAISS top-k + BM25 top-k, deduplicated by text.
        3. Cross-encoder reranks all candidates globally.
        4. Return top (top_k_per_company × n_companies) after reranking.
        """

        # ── 1. Encode query for FAISS ──────────────────────────────────────
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        tokenized_query = query.lower().split()

        # ── 2. Per-company retrieval ───────────────────────────────────────
        candidates: List[Chunk] = []
        seen_texts: set         = set()

        for company, (sub_index, sub_chunks) in self.company_indices.items():

            # FAISS
            k_faiss       = min(top_k_per_company, sub_index.ntotal)
            scores, idxs  = sub_index.search(q_emb, k_faiss)
            faiss_results = [sub_chunks[i] for i in idxs[0] if i >= 0]

            # BM25
            bm25_scores   = self.bm25_indices[company].get_scores(tokenized_query)
            k_bm25        = min(top_k_per_company, len(self.bm25_chunks[company]))
            top_bm25_idxs = bm25_scores.argsort()[-k_bm25:][::-1]
            bm25_results  = [self.bm25_chunks[company][i] for i in top_bm25_idxs]

            # Deduplicate — FAISS first (semantic signal), then BM25 additions
            for chunk in faiss_results + bm25_results:
                if chunk.text not in seen_texts:
                    seen_texts.add(chunk.text)
                    candidates.append(chunk)

        # ── 3. Rerank ──────────────────────────────────────────────────────
        if self.rerank and self.reranker is not None and candidates:
            pairs  = [(query, c.text) for c in candidates]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            ranked = sorted(
                zip(scores, candidates),
                key=lambda x: x[0],
                reverse=True,
            )
            # Keep top (k × n_companies) after reranking
            n_keep    = top_k_per_company * len(self.company_indices)
            candidates = [c for _, c in ranked[:n_keep]]

        return candidates