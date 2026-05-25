import { For, Show } from "solid-js";

import type { Signal } from "../lib/api";
import {
  fmtDate,
  fmtSigned,
  guidanceColor,
  sentimentColor,
  toneColor,
} from "../lib/format";

export function SignalCard(props: { signal: Signal }) {
  const s = () => props.signal;

  return (
    <div class="rounded-xl border border-slate-800 bg-slate-900/40 p-6 backdrop-blur">
      <div class="flex items-baseline justify-between">
        <div>
          <div class="text-xs text-slate-500 uppercase tracking-wider">
            {fmtDate(s().date)}
          </div>
          <div class="text-2xl font-bold mt-1 tracking-tight">{s().ticker}</div>
        </div>
        <div class={`text-3xl font-mono font-semibold ${sentimentColor(s().sentiment_score)}`}>
          {fmtSigned(s().sentiment_score)}
        </div>
      </div>

      <div class="grid grid-cols-3 gap-3 mt-6">
        <div>
          <div class="text-[10px] uppercase tracking-wider text-slate-500">Tone</div>
          <div
            class={`mt-1 inline-flex text-xs font-medium px-2 py-0.5 rounded-full border ${toneColor(s().tone)}`}
          >
            {s().tone}
          </div>
        </div>
        <div>
          <div class="text-[10px] uppercase tracking-wider text-slate-500">Guidance</div>
          <div class={`mt-1 text-sm font-medium capitalize ${guidanceColor(s().guidance_direction)}`}>
            {s().guidance_direction}
          </div>
        </div>
        <div>
          <div class="text-[10px] uppercase tracking-wider text-slate-500">Earnings</div>
          <div class="mt-1 text-sm font-medium text-slate-200 capitalize">
            {s().earnings_framing.replace("_", " ")}
          </div>
        </div>
      </div>

      <Show when={(s().risk_flags ?? []).length > 0}>
        <div class="mt-6">
          <div class="text-[10px] uppercase tracking-wider text-slate-500 mb-2">
            Risk flags
          </div>
          <div class="flex flex-wrap gap-1.5">
            <For each={s().risk_flags}>
              {(r) => (
                <span class="text-xs px-2 py-0.5 rounded-full border border-rose-500/30 bg-rose-500/10 text-rose-300">
                  {r}
                </span>
              )}
            </For>
          </div>
        </div>
      </Show>

      <Show when={(s().key_themes ?? []).length > 0}>
        <div class="mt-5">
          <div class="text-[10px] uppercase tracking-wider text-slate-500 mb-2">
            Key themes
          </div>
          <div class="flex flex-wrap gap-1.5">
            <For each={s().key_themes}>
              {(t) => (
                <span class="text-xs px-2 py-0.5 rounded-md border border-slate-700 bg-slate-800/60 text-slate-300">
                  {t}
                </span>
              )}
            </For>
          </div>
        </div>
      </Show>

      <Show when={s().reasoning}>
        <p class="mt-6 text-sm text-slate-400 leading-relaxed border-t border-slate-800 pt-4">
          {s().reasoning}
        </p>
      </Show>
    </div>
  );
}
