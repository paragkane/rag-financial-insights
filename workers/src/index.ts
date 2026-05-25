import { Hono } from "hono";
import { health } from "./routes/health";
import { signals } from "./routes/signals";
import { backtest } from "./routes/backtest";
import { filings } from "./routes/filings";

const app = new Hono();

app.route("/api/health", health);
app.route("/api/signals", signals);
app.route("/api/backtest", backtest);
app.route("/api/filings", filings);

export default app;
