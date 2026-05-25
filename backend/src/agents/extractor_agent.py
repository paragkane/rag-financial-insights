"""Extractor agent — Claude with strict tool-use schema.

Reuses src.extraction.signal_extractor.FilingSignal so the output stays
backwards-compatible with the existing backtester.
"""

from __future__ import annotations

from src.agents.base import Agent, AgentContext
from src.extraction.signal_extractor import FilingSignal

SYSTEM = (
    "You are a quantitative financial analyst. Read the retrieved passages from a SEC filing "
    "and emit a structured alpha signal. Be conservative — do not infer what is not stated."
)

USER_TEMPLATE = """Filing: {filing_id}

Retrieved passages (most relevant first):

[SENTIMENT-RELEVANT]
{sentiment_chunks}

[GUIDANCE-RELEVANT]
{guidance_chunks}

[RISK-RELEVANT]
{risk_chunks}
{critic_feedback}
Call the emit_signal tool with the structured signal."""

SIGNAL_TOOL_SCHEMA = {
    "description": "Emit the structured alpha signal for this filing.",
    "input_schema": {
        "type": "object",
        "required": [
            "sentiment_score", "guidance_direction", "guidance_magnitude",
            "guidance_confidence", "risk_flags", "earnings_framing",
            "tone", "key_themes", "reasoning",
        ],
        "properties": {
            "sentiment_score":     {"type": "number", "minimum": -1, "maximum": 1},
            "guidance_direction":  {"type": "string", "enum": ["raised", "lowered", "maintained", "none"]},
            "guidance_magnitude":  {"type": "number", "minimum": 0},
            "guidance_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "risk_flags": {
                "type": "array",
                "items": {"type": "string", "enum": [
                    "liquidity", "competition", "macro", "regulatory",
                    "execution", "demand", "margin",
                ]},
            },
            "earnings_framing": {"type": "string", "enum": ["beat", "miss", "in-line", "not_mentioned"]},
            "tone":             {"type": "string", "enum": ["optimistic", "cautious", "neutral", "defensive"]},
            "key_themes":       {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "reasoning":        {"type": "string"},
        },
    },
}


def _join(chunks: list[str]) -> str:
    return "\n\n".join(f"- {c.strip()}" for c in chunks) or "(none)"


class ExtractorAgent(Agent):
    name = "extractor"

    def run(self, ctx: AgentContext, critic_feedback: str = "") -> AgentContext:
        ctx.log(self.name, "start", revision=ctx.revisions)

        feedback_block = (
            f"\nPrior critic feedback to address:\n{critic_feedback}\n"
            if critic_feedback else ""
        )

        user = USER_TEMPLATE.format(
            filing_id=ctx.filing_id,
            sentiment_chunks=_join(ctx.retrieved_chunks.get("sentiment", [])),
            guidance_chunks=_join(ctx.retrieved_chunks.get("guidance", [])),
            risk_chunks=_join(ctx.retrieved_chunks.get("risk", [])),
            critic_feedback=feedback_block,
        )

        raw = self.call_tool(
            system=SYSTEM,
            user=user,
            tool_name="emit_signal",
            tool_schema=SIGNAL_TOOL_SCHEMA,
            max_tokens=1024,
            temperature=0,
        )

        # Validate through the existing Pydantic model so the contract is
        # identical to whatever the legacy single-pass extractor produces.
        validated = FilingSignal(**raw).model_dump()
        ctx.extraction = validated
        ctx.log(self.name, "finish", sentiment_score=validated["sentiment_score"])
        return ctx
