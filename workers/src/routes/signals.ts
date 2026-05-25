import { Hono } from "hono";

export const signals = new Hono();

signals.get("/", (c) => c.json({ todo: "list signals from KV" }));
