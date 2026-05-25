"""Hybrid retriever — BM25 sparse + dense embeddings, fused with RRF.

Reciprocal Rank Fusion is robust to score-scale differences across the two
backends, so we don't need to tune a per-corpus weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.rag.chunker import Chunk
from src.rag.embedder import Embedder

RRF_K = 60  # standard RRF constant — Cormack et al. 2009


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        self.chunks = chunks
        self.embedder = embedder or Embedder()
        self._embeddings = self.embedder.fit([c.text for c in chunks])
        self._bm25 = self._build_bm25([c.text for c in chunks])

    @staticmethod
    def _build_bm25(corpus: list[str]):
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [c.lower().split() for c in corpus]
            return BM25Okapi(tokenized) if tokenized else None
        except Exception:
            return None

    def _dense_scores(self, query: str) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0)
        q = self.embedder.encode_query(query)
        return self._embeddings @ q

    def _sparse_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None or not self.chunks:
            return np.zeros(len(self.chunks))
        return np.asarray(self._bm25.get_scores(query.lower().split()))

    def search(self, query: str, k: int = 8) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        dense = self._dense_scores(query)
        sparse = self._sparse_scores(query)

        def _ranks(scores: np.ndarray) -> dict[int, int]:
            order = np.argsort(-scores)
            return {int(idx): rank + 1 for rank, idx in enumerate(order)}

        d_rank, s_rank = _ranks(dense), _ranks(sparse)
        fused = {
            i: 1.0 / (RRF_K + d_rank[i]) + 1.0 / (RRF_K + s_rank[i])
            for i in range(len(self.chunks))
        }

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [ScoredChunk(chunk=self.chunks[i], score=score) for i, score in ordered]
