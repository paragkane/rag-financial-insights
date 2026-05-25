import { createMemo, For } from "solid-js";

import type { Signal } from "../lib/api";

const W = 800;
const H = 220;
const PAD = 28;

interface Dot {
  x: number;
  y: number;
  s: Signal;
}

export function ReturnsChart(props: { signals: Signal[] }) {
  const points = createMemo<{ path: string; area: string; dots: Dot[] }>(() => {
    const ss = [...props.signals].sort((a, b) => a.date.localeCompare(b.date));
    if (ss.length === 0) return { path: "", area: "", dots: [] };

    const denom = Math.max(ss.length - 1, 1);
    const xs = ss.map((_, i) => PAD + (i * (W - 2 * PAD)) / denom);
    const ys = ss.map((s) => {
      const norm = (s.sentiment_score + 1) / 2; // map [-1,1] -> [0,1]
      return H - PAD - norm * (H - 2 * PAD);
    });

    const path = xs
      .map((x, i) => `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${ys[i].toFixed(1)}`)
      .join(" ");

    const area = `${path} L ${xs[xs.length - 1].toFixed(1)} ${H - PAD} L ${xs[0].toFixed(1)} ${H - PAD} Z`;

    const dots: Dot[] = xs.map((x, i) => ({ x, y: ys[i], s: ss[i] }));
    return { path, area, dots };
  });

  const midY = H / 2;

  return (
    <div class="rounded-xl border border-slate-800 bg-slate-900/40 p-4 backdrop-blur">
      <svg viewBox={`0 0 ${W} ${H}`} class="w-full h-auto" preserveAspectRatio="none">
        <defs>
          <linearGradient id="sentGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#6366f1" stop-opacity="0.4" />
            <stop offset="100%" stop-color="#6366f1" stop-opacity="0" />
          </linearGradient>
        </defs>

        <line
          x1={PAD}
          y1={midY}
          x2={W - PAD}
          y2={midY}
          stroke="#334155"
          stroke-dasharray="3,3"
        />
        <text x={PAD} y={midY - 4} fill="#475569" font-size="10">0.0</text>
        <text x={PAD} y={PAD + 8} fill="#475569" font-size="10">+1.0</text>
        <text x={PAD} y={H - PAD - 2} fill="#475569" font-size="10">-1.0</text>

        <path d={points().area} fill="url(#sentGrad)" />
        <path d={points().path} fill="none" stroke="#818cf8" stroke-width="2" />

        <For each={points().dots}>
          {(d) => (
            <g>
              <circle
                cx={d.x}
                cy={d.y}
                r="3.5"
                fill={d.s.sentiment_score >= 0 ? "#34d399" : "#fb7185"}
              />
              <title>
                {d.s.date}: {d.s.sentiment_score.toFixed(2)} · {d.s.tone}
              </title>
            </g>
          )}
        </For>
      </svg>
    </div>
  );
}
