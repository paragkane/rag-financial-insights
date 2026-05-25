"""Embedder with a sentence-transformers backend and a TF-IDF fallback.

We default to a small, fast model (all-MiniLM-L6-v2) when available so the
local dev loop stays cheap. The fallback keeps the pipeline runnable
without the heavy ML dependency installed (CI, low-resource boxes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return (mat / norms).astype(np.float32)


class Embedder:
    """Embeds strings into a unit-norm float32 matrix.

    Usage:
        e = Embedder()
        corpus_vecs = e.fit(corpus)     # (n, d), L2-normalized
        q_vec       = e.encode_query(q) # (d,), L2-normalized
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._st = None
        self._tfidf = None
        self._fitted = False
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(model_name)
        except Exception:
            self._st = None

    def _encode_st(self, texts: list[str]) -> "NDArray[np.float32]":
        vecs = self._st.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return vecs.astype(np.float32)

    def fit(self, corpus: list[str]) -> "NDArray[np.float32]":
        """Embed the corpus. In TF-IDF mode, also fits the vocabulary."""
        if not corpus:
            return np.zeros((0, 384), dtype=np.float32)
        if self._st is not None:
            self._fitted = True
            return self._encode_st(corpus)

        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
        mat = self._tfidf.fit_transform(corpus).toarray()
        self._fitted = True
        return _l2_normalize(mat)

    def encode_query(self, query: str) -> "NDArray[np.float32]":
        if self._st is not None:
            return self._encode_st([query])[0]
        if not self._fitted or self._tfidf is None:
            raise RuntimeError("Call fit(corpus) before encode_query() in TF-IDF mode")
        mat = self._tfidf.transform([query]).toarray()
        return _l2_normalize(mat)[0]
