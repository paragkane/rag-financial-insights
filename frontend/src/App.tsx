import { Router, Route, A, type RouteSectionProps } from "@solidjs/router";

import { Dashboard } from "./routes/Dashboard";
import { TickerDetail } from "./routes/TickerDetail";
import { BacktestExplorer } from "./routes/BacktestExplorer";

function Shell(props: RouteSectionProps) {
  return (
    <div class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <header class="border-b border-slate-800/80 backdrop-blur sticky top-0 z-20 bg-slate-950/80">
        <div class="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
          <A href="/" class="flex items-center gap-3 group">
            <div class="size-8 rounded-md bg-gradient-to-br from-indigo-500 to-fuchsia-500 grid place-items-center font-bold text-sm shadow-lg shadow-indigo-500/20">
              R
            </div>
            <div>
              <div class="text-sm font-semibold tracking-tight">RAG Financial Insights</div>
              <div class="text-[11px] text-slate-500 -mt-0.5">Multi-agent SEC signal engine</div>
            </div>
          </A>
          <nav class="flex items-center gap-1 text-sm">
            <A
              href="/"
              end
              class="px-3 py-1.5 rounded-md text-slate-400 hover:bg-slate-800/60 hover:text-slate-100 transition"
              activeClass="!bg-slate-800/80 !text-white"
            >
              Dashboard
            </A>
            <A
              href="/backtest"
              class="px-3 py-1.5 rounded-md text-slate-400 hover:bg-slate-800/60 hover:text-slate-100 transition"
              activeClass="!bg-slate-800/80 !text-white"
            >
              Backtest
            </A>
            <a
              href="https://github.com/paragkane/rag-financial-insights"
              target="_blank"
              rel="noopener"
              class="px-3 py-1.5 rounded-md hover:bg-slate-800/60 transition text-slate-400"
            >
              GitHub
            </a>
          </nav>
        </div>
      </header>

      <main class="mx-auto max-w-7xl px-6 py-10">{props.children}</main>

      <footer class="border-t border-slate-800/80 mt-16">
        <div class="mx-auto max-w-7xl px-6 py-6 text-xs text-slate-500 flex flex-col sm:flex-row gap-2 sm:justify-between">
          <div>Edge-deployed on Cloudflare Pages + Workers KV</div>
          <div>Claude Sonnet 4.6 · Pydantic · BM25 + dense · RRF fusion</div>
        </div>
      </footer>
    </div>
  );
}

export function App() {
  return (
    <Router root={Shell}>
      <Route path="/" component={Dashboard} />
      <Route path="/ticker/:ticker" component={TickerDetail} />
      <Route path="/backtest" component={BacktestExplorer} />
    </Router>
  );
}
