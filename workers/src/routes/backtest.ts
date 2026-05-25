import { Hono } from "hono";
import type { Bindings } from "../lib/bindings";

export const backtest = new Hono<{ Bindings: Bindings }>();

backtest.get("/:signal", (c) =>
  c.json({ todo: "backtest payload", signal: c.req.param("signal") }),
);
