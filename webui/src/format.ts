const SCALE: Array<[number, string]> = [
  [1e12, 'T'],
  [1e9, 'G'],
  [1e6, 'M'],
  [1e3, 'k'],
];

/** Scales a rate to k/M/G/T, e.g. `12.34 Mkeys/s`. */
export function formatRate(rate: number, unit: string): string {
  if (!Number.isFinite(rate) || rate <= 0) return `0 ${unit}`;
  for (const [factor, prefix] of SCALE) {
    if (rate >= factor) return `${(rate / factor).toFixed(2)} ${prefix}${unit}`;
  }
  return `${rate.toFixed(2)} ${unit}`;
}

/** Work amount as a power of two, matching the CLI (`2^34.10`). */
export function formatPow2(n: number | null): string {
  if (n === null || !Number.isFinite(n)) return '—';
  if (n <= 0) return '0';
  return `2^${Math.log2(n).toFixed(2)}`;
}

const YEAR = 365.25 * 86400;

/** `hh:mm:ss`, `Nd hh:mm:ss`, or years for very long durations. */
export function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return '—';
  if (seconds >= 1000 * YEAR) return `${(seconds / YEAR).toExponential(2)} years`;
  if (seconds >= YEAR) return `${(seconds / YEAR).toFixed(1)} years`;
  const s = Math.floor(seconds);
  const days = Math.floor(s / 86400);
  const hms = [Math.floor((s % 86400) / 3600), Math.floor((s % 3600) / 60), s % 60]
    .map((v) => String(v).padStart(2, '0'))
    .join(':');
  return days > 0 ? `${days}d ${hms}` : hms;
}

export function formatTime(iso: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/** Strips leading zeros from a hex key for compact display (keeps at least one digit). */
export function shortHex(hex: string | null): string {
  if (!hex) return '—';
  return hex.replace(/^0+(?=.)/, '');
}
