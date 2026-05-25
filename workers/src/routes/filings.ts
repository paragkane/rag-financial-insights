import { Hono } from "hono";
import type { Bindings } from "../lib/bindings";

export const filings = new Hono<{ Bindings: Bindings }>();

filings.get("/", (c) => c.json({ todo: "list filings" }));
