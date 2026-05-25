import { Hono } from "hono";

export const filings = new Hono();

filings.get("/", (c) => c.json({ todo: "list filings" }));
