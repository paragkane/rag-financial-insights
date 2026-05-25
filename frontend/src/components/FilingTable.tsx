import { For } from "solid-js";

import type { Signal } from "../lib/api";
import { fmtDate, fmtSigned, guidanceColor, sentimentColor } from "../lib/format";

export function FilingTable(props: { signals: Signal[] }) {
  const sorted = () =>
    [...props.signals].sort((a, b) => b.date.localeCompare(a.date));

  return (
    <div class="rounded-xl border border-slate-800 overflow-hidden">
      <table class="w-full text-sm">
        <thead class="bg-slate-900/60 text-xs uppercase tracking-wider text-slate-400">
          <tr>
            <th class="text-left  px-4 py-3">Date</th>
            <th class="text-right px-4 py-3">Sentiment</th>
            <th class="text-left  px-4 py-3">Tone</th>
            <th class="text-left  px-4 py-3">Guidance</th>
            <th class="text-left  px-4 py-3">Earnings</th>
            <th class="text-left  px-4 py-3">Risk flags</th>
          </tr>
        </thead>
        <tbody>
          <For each={sorted()}>
            {(s) => (
              <tr class="border-t border-slate-800/80 hover:bg-slate-900/40">
                <td class="px-4 py-3 text-slate-300 whitespace-nowrap">
                  {fmtDate(s.date)}
                </td>
                <td class={`px-4 py-3 text-right font-mono ${sentimentColor(s.sentiment_score)}`}>
                  {fmtSigned(s.sentiment_score)}
                </td>
                <td class="px-4 py-3 text-slate-200 capitalize">{s.tone}</td>
                <td class={`px-4 py-3 capitalize ${guidanceColor(s.guidance_direction)}`}>
                  {s.guidance_direction}
                </td>
                <td class="px-4 py-3 text-slate-300 capitalize">
                  {s.earnings_framing.replace("_", " ")}
                </td>
                <td class="px-4 py-3 text-slate-400 text-xs">
                  {(s.risk_flags ?? []).join(", ") || "—"}
                </td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );
}
