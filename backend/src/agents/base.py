"""Multi-agent orchestration primitives.

AgentContext threads state through Retriever -> Extractor -> Critic -> Reconciler.
Every agent reads from and writes to the same context, leaving a full audit
trail under data/processed/agent_traces/<filing_id>.json.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"

TRACE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "agent_traces"


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


@dataclass
class TraceEvent:
    agent: str
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class AgentContext:
    """Shared state for one filing pass through the agent pipeline."""

    filing_id: str
    ticker: str
    filing_text: str
    retrieved_chunks: dict[str, list[str]] = field(default_factory=dict)
    extraction: dict[str, Any] | None = None
    critique: dict[str, Any] | None = None
    final_signal: dict[str, Any] | None = None
    revisions: int = 0
    trace: list[TraceEvent] = field(default_factory=list)

    def log(self, agent: str, event: str, **payload: Any) -> None:
        self.trace.append(TraceEvent(agent=agent, event=event, payload=payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "filing_id": self.filing_id,
            "ticker": self.ticker,
            "retrieved_chunks": self.retrieved_chunks,
            "extraction": self.extraction,
            "critique": self.critique,
            "final_signal": self.final_signal,
            "revisions": self.revisions,
            "trace": [
                {"agent": t.agent, "event": t.event, "payload": t.payload, "ts": t.ts}
                for t in self.trace
            ],
        }

    def persist_trace(self) -> Path:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / f"{self.filing_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


class Agent(ABC):
    """Base class for every agent in the pipeline."""

    name: str = "agent"
    model: str = DEFAULT_MODEL

    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = get_client()
        return self._client

    @abstractmethod
    def run(self, ctx: AgentContext) -> AgentContext: ...

    def call_tool(
        self,
        *,
        system: str,
        user: str,
        tool_name: str,
        tool_schema: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Strict structured output via Anthropic tool use.

        The model is forced to call `tool_name` and we return its input dict.
        Any non-tool reply raises — keeps the pipeline deterministic.
        """
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            tools=[{
                "name": tool_name,
                "description": tool_schema.get("description", ""),
                "input_schema": tool_schema["input_schema"],
            }],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        for block in msg.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input  # type: ignore[return-value]
        raise RuntimeError(f"{tool_name} did not return a tool_use block")
