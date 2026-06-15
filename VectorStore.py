"""
VectorStore.py
==============
Hybrid retrieval: FAISS (semantic) + BM25 (keyword) with cross-encoder reranking.

Class
-----
VectorStore
    build()    : embed chunks, build FAISS + BM25 indices per company
    retrieve() : FAISS top-k + BM25 top-k → deduplicate → rerank → return top-40
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
    Retrieval pipeline: FAISS + BM25 per company, then cross-encoder reranking.

    Per-company sub-indices guarantee representation from each company regardless
    of corpus size imbalance (JFE has 311 pages, JSW has 18).

    FAISS handles semantic similarity; BM25 catches exact token matches
    (numbers, fiscal year labels) that embedding models can miss.
    The cross-encoder reranks up to 80 candidates jointly and trims to 40.
    """

    def __init__(self, model_name: str = EMBED_MODEL,
                 rerank_model: str = RERANK_MODEL, rerank: bool = True):
        print(f"Loading embedding model: {model_name} …")
        self.model = SentenceTransformer(model_name)

        self.rerank = rerank
        if rerank:
            print(f"Loading reranker: {rerank_model} …")
            self.reranker = CrossEncoder(rerank_model)
        else:
            self.reranker = None

        self.index:           Optional[faiss.IndexFlatIP]                      = None
        self.chunks:          List[Chunk]                                       = []
        self.company_indices: Dict[str, Tuple[faiss.IndexFlatIP, List[Chunk]]] = {}
        self.bm25_indices:    Dict[str, BM25Okapi]                             = {}
        self.bm25_chunks:     Dict[str, List[Chunk]]                           = {}

    def build(self, chunks: List[Chunk], batch_size: int = 256) -> None:
        """Embed all chunks; build full FAISS index, per-company FAISS sub-indices, and BM25 indices."""
        texts = [c.text for c in chunks]
        print(f"Embedding {len(texts)} chunks …")
        embeddings = self.model.encode(texts, batch_size=batch_size,
                                       show_progress_bar=True,
                                       normalize_embeddings=True,
                                       convert_to_numpy=True)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks
        print(f"Full FAISS index: {self.index.ntotal} vectors, dim={dim}")

        buckets: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            buckets[c.company].append(i)

        for company, positions in buckets.items():
            company_chunks = [chunks[i] for i in positions]

            sub_embs  = embeddings[positions].astype(np.float32)
            sub_index = faiss.IndexFlatIP(dim)
            sub_index.add(sub_embs)
            self.company_indices[company] = (sub_index, company_chunks)

            tokenized = [c.text.lower().split() for c in company_chunks]
            self.bm25_indices[company] = BM25Okapi(tokenized)
            self.bm25_chunks[company]  = company_chunks

            print(f"  [{company}] FAISS: {sub_index.ntotal} vectors | BM25: {len(tokenized)} docs")

    def retrieve(self, query: str, top_k_per_company: int = TOP_K_PER_COMPANY) -> List[Chunk]:
        """FAISS + BM25 retrieval per company → deduplicate → cross-encoder rerank → top-40."""
        q_emb = self.model.encode([query], normalize_embeddings=True,
                                  convert_to_numpy=True).astype(np.float32)
        tokenized_query = query.lower().split()

        candidates: List[Chunk] = []
        seen_texts: set         = set()

        for company, (sub_index, sub_chunks) in self.company_indices.items():
            # FAISS
            k             = min(top_k_per_company, sub_index.ntotal)
            _, idxs       = sub_index.search(q_emb, k)
            faiss_results = [sub_chunks[i] for i in idxs[0] if i >= 0]

            # BM25
            bm25_scores   = self.bm25_indices[company].get_scores(tokenized_query)
            k_bm25        = min(top_k_per_company, len(self.bm25_chunks[company]))
            top_idxs      = bm25_scores.argsort()[-k_bm25:][::-1]
            bm25_results  = [self.bm25_chunks[company][i] for i in top_idxs]

            for chunk in faiss_results + bm25_results:
                if chunk.text not in seen_texts:
                    seen_texts.add(chunk.text)
                    candidates.append(chunk)

        if self.rerank and self.reranker and candidates:
            scores = self.reranker.predict([(query, c.text) for c in candidates],
                                           show_progress_bar=False)
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            n_keep = top_k_per_company * len(self.company_indices)
            candidates = [c for _, c in ranked[:n_keep]]

        return candidates
