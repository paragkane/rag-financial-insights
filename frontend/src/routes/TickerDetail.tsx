import { useParams } from "@solidjs/router";
import { createResource, Show } from "solid-js";

import { api } from "../lib/api";
import { FilingTable } from "../components/FilingTable";
import { ReturnsChart } from "../components/ReturnsChart";
import { SignalCard } from "../components/SignalCard";

export function TickerDetail() {
  const params = useParams<{ ticker: string }>();
  const [signals] = createResource(
    () => params.ticker,
    (t) => api.getSignals(t).catch(() => []),
  );

  return (
    <div class="space-y-10">
      <div>
        <a href="/" class="text-xs text-slate-500 hover:text-slate-300">
          ← Back
        </a>
        <h1 class="text-3xl font-bold mt-1 tracking-tight">{params.ticker}</h1>
        <Show when={signals()}>
          <p class="text-sm text-slate-500 mt-1">
            {signals()!.length} filings
            <Show when={signals()!.length > 0}>
              {" "}
              · latest {signals()![signals()!.length - 1]?.date}
            </Show>
          </p>
        </Show>
      </div>

      <Show
        when={!signals.loading}
        fallback={<div class="text-slate-500 text-sm">Loading from edge…</div>}
      >
        <Show
          when={(signals() ?? []).length > 0}
          fallback={
            <div class="rounded-xl border border-dashed border-slate-800 p-8 text-center text-slate-500 text-sm">
              No signals available for {params.ticker}.
            </div>
          }
        >
          <section>
            <h2 class="text-sm font-semibold uppercase tracking-widest text-slate-400 mb-3">
              Sentiment over time
            </h2>
            <ReturnsChart signals={signals()!} />
          </section>

          <section>
            <h2 class="text-sm font-semibold uppercase tracking-widest text-slate-400 mb-3">
              Latest filing
            </h2>
            <SignalCard signal={signals()![signals()!.length - 1]} />
          </section>

          <section>
            <h2 class="text-sm font-semibold uppercase tracking-widest text-slate-400 mb-3">
              All filings
            </h2>
            <FilingTable signals={signals()!} />
          </section>
        </Show>
      </Show>
    </div>
  );
}
