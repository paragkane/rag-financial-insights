import { Hono } from "hono";
import type { Bindings } from "../lib/bindings";

export const health = new Hono<{ Bindings: Bindings }>();

health.get("/", (c) =>
  c.json({ status: "ok", kv: Boolean(c.env.SIGNALS) }),
);
