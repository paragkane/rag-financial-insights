"""Reconciler — orchestrates the full Retriever -> Extractor -> Critic loop.

If critic confidence < threshold (or verdict == "revise"), inject the critic
feedback into the extractor and re-run, up to max_revisions times.
"""

from __future__ import annotations

from src.agents.base import Agent, AgentContext
from src.agents.critic_agent import CriticAgent
from src.agents.extractor_agent import ExtractorAgent
from src.agents.retriever_agent import RetrieverAgent

CONFIDENCE_THRESHOLD = 0.7
MAX_REVISIONS = 2


def _format_feedback(critique: dict | None) -> str:
    if not critique or not critique.get("issues"):
        return ""
    return "\n".join(
        f"- [{i['severity']}] {i['field']}: {i['note']}" for i in critique["issues"]
    )


class Reconciler(Agent):
    name = "reconciler"

    def __init__(
        self,
        retriever: RetrieverAgent | None = None,
        extractor: ExtractorAgent | None = None,
        critic: CriticAgent | None = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_revisions: int = MAX_REVISIONS,
    ):
        super().__init__()
        self.retriever = retriever or RetrieverAgent()
        self.extractor = extractor or ExtractorAgent()
        self.critic = critic or CriticAgent()
        self.confidence_threshold = confidence_threshold
        self.max_revisions = max_revisions

    def run(self, ctx: AgentContext) -> AgentContext:
        ctx.log(self.name, "start")

        # 1. Retrieve once — chunks don't change between revisions.
        self.retriever.run(ctx)

        # 2. First extraction + critique.
        self.extractor.run(ctx)
        self.critic.run(ctx)

        # 3. Revision loop — gated by critic verdict + confidence.
        while (
            ctx.revisions < self.max_revisions
            and ctx.critique is not None
            and (
                ctx.critique["verdict"] == "revise"
                or ctx.critique["confidence"] < self.confidence_threshold
            )
        ):
            ctx.revisions += 1
            feedback = _format_feedback(ctx.critique)
            ctx.log(self.name, "revise", revision=ctx.revisions, feedback=feedback)
            self.extractor.run(ctx, critic_feedback=feedback)
            self.critic.run(ctx)

        ctx.final_signal = ctx.extraction
        ctx.log(
            self.name, "finish",
            revisions=ctx.revisions,
            final_confidence=(ctx.critique or {}).get("confidence"),
        )
        ctx.persist_trace()
        return ctx


def run_pipeline(filing_id: str, ticker: str, filing_text: str) -> AgentContext:
    """Convenience entrypoint — one filing in, fully-processed ctx out."""
    ctx = AgentContext(filing_id=filing_id, ticker=ticker, filing_text=filing_text)
    return Reconciler().run(ctx)
