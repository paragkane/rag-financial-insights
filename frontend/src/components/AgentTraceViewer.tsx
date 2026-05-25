import { For } from "solid-js";

interface AgentStep {
  name: string;
  desc: string;
  color: string;
}

const AGENTS: AgentStep[] = [
  {
    name: "Retriever",
    desc: "Section-aware chunker · BM25 + dense embeddings · RRF fusion",
    color: "from-sky-500 to-cyan-500",
  },
  {
    name: "Extractor",
    desc: "Claude Sonnet 4.6 tool-use → strict Pydantic schema",
    color: "from-indigo-500 to-violet-500",
  },
  {
    name: "Critic",
    desc: "Audits extraction against retrieved passages · returns confidence",
    color: "from-fuchsia-500 to-pink-500",
  },
  {
    name: "Reconciler",
    desc: "Re-runs extractor with critic feedback if confidence < 0.7",
    color: "from-amber-500 to-orange-500",
  },
];

export function AgentTraceViewer() {
  return (
    <div class="rounded-xl border border-slate-800 bg-slate-900/40 p-6 backdrop-blur">
      <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
        <For each={AGENTS}>
          {(a, i) => (
            <div class="relative">
              <div class="rounded-lg border border-slate-800 bg-slate-950/40 p-4 h-full">
                <div class="flex items-center gap-2">
                  <div
                    class={`size-6 rounded-md bg-gradient-to-br ${a.color} grid place-items-center text-[10px] font-bold text-white`}
                  >
                    {i() + 1}
                  </div>
                  <div class="text-sm font-semibold">{a.name}</div>
                </div>
                <p class="text-xs text-slate-400 mt-2 leading-relaxed">{a.desc}</p>
              </div>
              {i() < AGENTS.length - 1 && (
                <div class="hidden md:block absolute top-1/2 -right-2 -translate-y-1/2 text-slate-700">
                  →
                </div>
              )}
            </div>
          )}
        </For>
      </div>
      <div class="mt-5 text-xs text-slate-500 leading-relaxed">
        All four agents share an{" "}
        <code class="text-slate-300 bg-slate-900 px-1 py-0.5 rounded">AgentContext</code>{" "}
        with full trace persisted to{" "}
        <code class="text-slate-300 bg-slate-900 px-1 py-0.5 rounded">
          data/processed/agent_traces/&lt;filing_id&gt;.json
        </code>
        .
      </div>
    </div>
  );
}
