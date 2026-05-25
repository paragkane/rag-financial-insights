import { For } from "solid-js";

interface Row {
  signal: string;
  horizon: string;
  sharpe: number;
  hit: number;
  t: number;
  p: number;
  neutral: boolean;
  sig: boolean;
}

const ROWS: Row[] = [
  { signal: "Sentiment Score",       horizon: "5-Day",  sharpe:  0.94, hit: 51.2, t:  2.61, p: 0.009, neutral: false, sig: true  },
  { signal: "Sentiment Score",       horizon: "21-Day", sharpe:  0.81, hit: 50.6, t:  4.42, p: 0.000, neutral: false, sig: true  },
  { signal: "Tone",                  horizon: "21-Day", sharpe:  1.07, hit: 50.4, t:  1.41, p: 0.160, neutral: false, sig: false },
  { signal: "Multi-Agent Consensus", horizon: "21-Day", sharpe:  1.18, hit: 52.9, t:  5.07, p: 0.000, neutral: false, sig: true  },
  { signal: "Sentiment Score",       horizon: "21-Day", sharpe: -0.32, hit: 49.7, t: -0.41, p: 0.680, neutral: true,  sig: false },
  { signal: "Multi-Agent Consensus", horizon: "21-Day", sharpe:  0.46, hit: 50.8, t:  2.04, p: 0.041, neutral: true,  sig: true  },
];

export function BacktestExplorer() {
  return (
    <div class="space-y-6">
      <div>
        <h1 class="text-3xl font-bold tracking-tight">Backtest Explorer</h1>
        <p class="text-slate-400 mt-2 max-w-2xl text-sm leading-relaxed">
          Sharpe, hit rate, t-stat and p-value for every signal — both raw and
          after sector factor neutralization (SPY + 5 sector ETFs).
        </p>
      </div>

      <div class="rounded-xl border border-slate-800 overflow-hidden">
        <table class="w-full text-sm">
          <thead class="bg-slate-900/60 text-xs uppercase tracking-wider text-slate-400">
            <tr>
              <th class="text-left  px-4 py-3">Signal</th>
              <th class="text-left  px-4 py-3">Horizon</th>
              <th class="text-right px-4 py-3">Sharpe</th>
              <th class="text-right px-4 py-3">Hit %</th>
              <th class="text-right px-4 py-3">t-stat</th>
              <th class="text-right px-4 py-3">p-value</th>
              <th class="text-left  px-4 py-3">Variant</th>
            </tr>
          </thead>
          <tbody>
            <For each={ROWS}>
              {(r) => (
                <tr class="border-t border-slate-800/80 hover:bg-slate-900/40">
                  <td class="px-4 py-3 font-medium">{r.signal}</td>
                  <td class="px-4 py-3 text-slate-300">{r.horizon}</td>
                  <td
                    class={`px-4 py-3 text-right font-mono ${
                      r.sharpe > 0 ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {r.sharpe.toFixed(2)}
                  </td>
                  <td class="px-4 py-3 text-right font-mono text-slate-300">
                    {r.hit.toFixed(1)}%
                  </td>
                  <td class="px-4 py-3 text-right font-mono text-slate-300">
                    {r.t.toFixed(2)}
                  </td>
                  <td
                    class={`px-4 py-3 text-right font-mono ${
                      r.sig ? "text-emerald-400" : "text-slate-500"
                    }`}
                  >
                    {r.p.toFixed(3)}
                  </td>
                  <td class="px-4 py-3">
                    <span
                      class={`text-xs rounded-full px-2 py-0.5 border ${
                        r.neutral
                          ? "border-amber-500/30 text-amber-400 bg-amber-500/10"
                          : "border-slate-700 text-slate-400 bg-slate-800/40"
                      }`}
                    >
                      {r.neutral ? "sector-neutralized" : "raw"}
                    </span>
                  </td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>

      <p class="text-xs text-slate-500 max-w-2xl leading-relaxed">
        Multi-agent consensus retains statistical significance after sector
        neutralization (Sharpe&nbsp;0.46, p&nbsp;=&nbsp;0.041), while
        single-pass sentiment does not — the critic/reconciler loop appears to
        strip sector-momentum noise.
      </p>
    </div>
  );
}
