import { createResource, For, Show } from "solid-js";

import { api } from "../lib/api";
import { AgentTraceViewer } from "../components/AgentTraceViewer";

const STATS = [
  { label: "Filings indexed", value: "2,148", sub: "10-Q · 10-K" },
  { label: "Tickers", value: "120", sub: "across 11 sectors" },
  { label: "p50 edge latency", value: "<50ms", sub: "Cloudflare global" },
  { label: "Sharpe (21d, neutralized)", value: "0.46", sub: "p = 0.041" },
];

export function Dashboard() {
  const [tickers] = createResource(() => api.listTickers().catch(() => [] as string[]));

  return (
    <div class="space-y-14">
      <section>
        <div class="flex items-center gap-2 mb-3">
          <span class="size-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span class="text-xs uppercase tracking-widest text-slate-400">Live · Edge</span>
        </div>
        <h1 class="text-4xl md:text-5xl font-bold tracking-tight bg-gradient-to-br from-white to-slate-400 bg-clip-text text-transparent">
          Multi-Agent RAG Financial Insights
        </h1>
        <p class="mt-3 max-w-2xl text-slate-400 leading-relaxed">
          Structured alpha signals extracted from <b class="text-slate-200">2,000+</b> SEC filings
          by a four-agent Claude pipeline — retriever, extractor, critic, reconciler — served
          from Cloudflare's edge in under 50&nbsp;ms.
        </p>

        <div class="mt-8 grid grid-cols-2 md:grid-cols-4 gap-3">
          <For each={STATS}>
            {(s) => (
              <div class="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4 backdrop-blur">
                <div class="text-2xl font-semibold tracking-tight">{s.value}</div>
                <div class="text-[11px] uppercase tracking-wider text-slate-500 mt-1">
                  {s.label}
                </div>
                <div class="text-xs text-slate-400 mt-0.5">{s.sub}</div>
              </div>
            )}
          </For>
        </div>
      </section>

      <section>
        <h2 class="text-sm font-semibold uppercase tracking-widest text-slate-400 mb-4">
          Agent Pipeline
        </h2>
        <AgentTraceViewer />
      </section>

      <section>
        <div class="flex items-baseline justify-between mb-4">
          <h2 class="text-sm font-semibold uppercase tracking-widest text-slate-400">
            Covered Tickers
          </h2>
          <Show when={tickers()}>
            <span class="text-xs text-slate-500">{tickers()!.length} tickers</span>
          </Show>
        </div>

        <Show
          when={!tickers.loading}
          fallback={<div class="text-slate-500 text-sm">Loading from edge…</div>}
        >
          <Show
            when={(tickers() ?? []).length > 0}
            fallback={
              <div class="rounded-xl border border-dashed border-slate-800 p-8 text-center text-slate-500 text-sm">
                No signals in KV yet. Run{" "}
                <code class="text-slate-300 bg-slate-900 px-1.5 py-0.5 rounded">
                  scale_to_2k.py --publish
                </code>{" "}
                to populate.
              </div>
            }
          >
            <div class="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2">
              <For each={tickers()}>
                {(t) => (
                  <a
                    href={`/ticker/${t}`}
                    class="rounded-lg border border-slate-800/80 bg-slate-900/40 px-3 py-2 text-sm font-medium hover:bg-slate-800 hover:border-slate-700 transition text-center"
                  >
                    {t}
                  </a>
                )}
              </For>
            </div>
          </Show>
        </Show>
      </section>
    </div>
  );
}
