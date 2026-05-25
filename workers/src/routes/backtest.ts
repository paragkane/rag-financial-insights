import { Hono } from "hono";

export const backtest = new Hono();

backtest.get("/:signal", (c) =>
  c.json({ todo: "backtest payload", signal: c.req.param("signal") }),
);
