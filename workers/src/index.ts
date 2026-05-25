import { Hono } from "hono";
import { cors } from "hono/cors";

import type { Bindings } from "./lib/bindings";
import { health } from "./routes/health";
import { signals } from "./routes/signals";
import { backtest } from "./routes/backtest";
import { filings } from "./routes/filings";

const app = new Hono<{ Bindings: Bindings }>();

// SolidJS dashboard on Pages calls this Worker from a different origin.
app.use("*", cors());

app.route("/api/health", health);
app.route("/api/signals", signals);
app.route("/api/backtest", backtest);
app.route("/api/filings", filings);

export default app;
