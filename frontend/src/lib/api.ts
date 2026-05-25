// Typed client for the Cloudflare Workers API.
// Reads VITE_API_BASE at build time so prod points at the deployed Worker
// and dev defaults to wrangler's local server.

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8787";

export type Tone = "optimistic" | "cautious" | "neutral" | "defensive";
export type GuidanceDirection = "raised" | "lowered" | "maintained" | "none";
export type EarningsFraming = "beat" | "miss" | "in-line" | "not_mentioned";

export interface Signal {
  ticker: string;
  date: string;
  sentiment_score: number;
  guidance_direction: GuidanceDirection;
  guidance_magnitude?: number;
  guidance_confidence?: number;
  risk_flags: string[];
  earnings_framing: EarningsFraming;
  tone: Tone;
  key_themes: string[];
  reasoning: string;
}

export interface BacktestResult {
  signal: string;
  horizon_days: 1 | 5 | 21;
  sharpe: number;
  hit_rate: number;
  t_stat: number;
  p_value: number;
  n: number;
  sector_neutralized?: boolean;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => jsonFetch<{ status: string; kv: boolean }>("/api/health"),

  listTickers: async (): Promise<string[]> => {
    const r = await jsonFetch<{ tickers: string[] }>("/api/signals");
    return r.tickers ?? [];
  },

  getSignals: async (ticker: string): Promise<Signal[]> => {
    const r = await jsonFetch<{ ticker: string; signals: Signal[] }>(
      `/api/signals/${encodeURIComponent(ticker)}`,
    );
    return r.signals ?? [];
  },

  getBacktest: (signal: string): Promise<BacktestResult> =>
    jsonFetch<BacktestResult>(`/api/backtest/${encodeURIComponent(signal)}`),
};

/**
 * SSE subscriber stub — the Worker route /api/signals/stream isn't shipped
 * yet, but components can call this and degrade gracefully if it 404s.
 * Returns an unsubscribe function.
 */
export function subscribeToSignals(onSignal: (s: Signal) => void): () => void {
  let es: EventSource | null = null;
  try {
    es = new EventSource(`${API_BASE}/api/signals/stream`);
    es.onmessage = (e) => {
      try {
        onSignal(JSON.parse(e.data) as Signal);
      } catch {
        /* malformed payload — ignore */
      }
    };
    es.onerror = () => es?.close();
  } catch {
    /* server doesn't support streaming yet */
  }
  return () => es?.close();
}
