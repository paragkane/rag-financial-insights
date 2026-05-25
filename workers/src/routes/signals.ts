import { Hono } from "hono";
import type { Bindings } from "../lib/bindings";

export const signals = new Hono<{ Bindings: Bindings }>();

/**
 * GET /api/signals
 *   - no params      -> { tickers: ["AAPL", ...] }
 *   - ?ticker=AAPL   -> { ticker, signals: [...] }
 */
signals.get("/", async (c) => {
  const ticker = c.req.query("ticker");
  if (ticker) {
    const rows = await c.env.SIGNALS.get(`signals:${ticker.toUpperCase()}`, "json");
    if (!rows) return c.json({ error: "not found", ticker }, 404);
    return c.json({ ticker: ticker.toUpperCase(), signals: rows });
  }
  const index = await c.env.SIGNALS.get<string[]>("signals:index", "json");
  return c.json({ tickers: index ?? [] });
});

/** GET /api/signals/:ticker  — REST-style alternative */
signals.get("/:ticker", async (c) => {
  const ticker = c.req.param("ticker").toUpperCase();
  const rows = await c.env.SIGNALS.get(`signals:${ticker}`, "json");
  if (!rows) return c.json({ error: "not found", ticker }, 404);
  return c.json({ ticker, signals: rows });
});
