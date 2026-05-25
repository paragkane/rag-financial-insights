export const fmtPct = (n: number, digits = 1): string =>
  `${(n * 100).toFixed(digits)}%`;

export const fmtSigned = (n: number, digits = 2): string =>
  `${n >= 0 ? "+" : ""}${n.toFixed(digits)}`;

export const fmtDate = (iso: string): string => {
  try {
    return new Date(iso).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
};

const TONE_COLORS: Record<string, string> = {
  optimistic: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  cautious: "text-amber-400 bg-amber-500/10 border-amber-500/30",
  neutral: "text-slate-300 bg-slate-500/10 border-slate-500/30",
  defensive: "text-rose-400 bg-rose-500/10 border-rose-500/30",
};

export const toneColor = (tone: string): string =>
  TONE_COLORS[tone] ?? "text-slate-300 bg-slate-500/10 border-slate-500/30";

const GUIDANCE_COLORS: Record<string, string> = {
  raised: "text-emerald-400",
  lowered: "text-rose-400",
  maintained: "text-slate-300",
  none: "text-slate-500",
};

export const guidanceColor = (g: string): string =>
  GUIDANCE_COLORS[g] ?? "text-slate-400";

export const sentimentColor = (n: number): string =>
  n > 0.15 ? "text-emerald-400" : n < -0.15 ? "text-rose-400" : "text-slate-300";
