"""Critic agent — second Claude pass that audits the extraction against the
retrieved passages and returns a fidelity score + a list of issues.
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.base import Agent, AgentContext

SYSTEM = (
    "You are a critic auditing a structured signal extracted from a SEC filing. "
    "Compare the extraction against the retrieved passages and flag any field that "
    "is unsupported, hallucinated, or inconsistent with the source. Be strict but fair."
)

USER_TEMPLATE = """Filing: {filing_id}

Retrieved passages (the only ground truth):
{all_chunks}

Extraction to audit:
{extraction_json}

Call the emit_critique tool."""

CRITIQUE_TOOL_SCHEMA = {
    "description": "Score the extraction for fidelity to the retrieved passages.",
    "input_schema": {
        "type": "object",
        "required": ["confidence", "issues", "verdict"],
        "properties": {
            "confidence": {
                "type": "number", "minimum": 0, "maximum": 1,
                "description": "Overall confidence that the extraction is faithful to the source. 1.0 = perfect, 0 = wholly unsupported.",
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["field", "severity", "note"],
                    "properties": {
                        "field":    {"type": "string"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "note":     {"type": "string"},
                    },
                },
            },
            "verdict": {"type": "string", "enum": ["accept", "revise"]},
        },
    },
}


class CriticAgent(Agent):
    name = "critic"

    def run(self, ctx: AgentContext) -> AgentContext:
        if ctx.extraction is None:
            raise RuntimeError("CriticAgent requires ctx.extraction (run ExtractorAgent first)")

        ctx.log(self.name, "start", revision=ctx.revisions)

        all_chunks_text = ""
        for cls, chunks in ctx.retrieved_chunks.items():
            joined = "\n".join(f"  - {c.strip()}" for c in chunks)
            all_chunks_text += f"\n[{cls.upper()}]\n{joined}\n"

        user = USER_TEMPLATE.format(
            filing_id=ctx.filing_id,
            all_chunks=all_chunks_text or "(none)",
            extraction_json=json.dumps(ctx.extraction, indent=2),
        )

        critique: dict[str, Any] = self.call_tool(
            system=SYSTEM,
            user=user,
            tool_name="emit_critique",
            tool_schema=CRITIQUE_TOOL_SCHEMA,
            max_tokens=1024,
            temperature=0,
        )
        ctx.critique = critique
        ctx.log(self.name, "finish",
                confidence=critique["confidence"], verdict=critique["verdict"])
        return ctx
