"""Retriever agent — chunks the filing, builds a hybrid index, and pulls the
top passages for each signal class.

Output lands in ctx.retrieved_chunks as:
    {"sentiment": [...], "guidance": [...], "risk": [...]}
"""

from __future__ import annotations

from src.agents.base import Agent, AgentContext
from src.rag.chunker import chunk_filing
from src.rag.embedder import Embedder
from src.rag.hybrid_retriever import HybridRetriever

QUERIES = {
    "sentiment": "management tone, outlook, optimism, caution, confidence in business",
    "guidance":  "forward guidance, outlook raised lowered maintained revenue earnings forecast",
    "risk":      "risk factors, liquidity, macro headwinds, competition, regulatory exposure",
}

TOP_K = 8


class RetrieverAgent(Agent):
    name = "retriever"

    def __init__(self, embedder: Embedder | None = None):
        super().__init__()
        self.embedder = embedder or Embedder()

    def run(self, ctx: AgentContext) -> AgentContext:
        ctx.log(self.name, "start", chars=len(ctx.filing_text))

        chunks = chunk_filing(ctx.filing_text)
        retriever = HybridRetriever(chunks, embedder=self.embedder)

        for signal_class, query in QUERIES.items():
            top = retriever.search(query, k=TOP_K)
            ctx.retrieved_chunks[signal_class] = [c.text for c in top]
            ctx.log(self.name, "retrieved", signal_class=signal_class, n=len(top))

        ctx.log(self.name, "finish")
        return ctx
