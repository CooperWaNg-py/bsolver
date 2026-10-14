import { useState } from 'react';
import type { BenchResult, JobState } from '../types';
import { formatRate } from '../format';

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // Plain-HTTP remote access has no Clipboard API; fall back to a hidden textarea.
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try {
    if (!document.execCommand('copy')) throw new Error('copy failed');
  } finally {
    document.body.removeChild(ta);
  }
}

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [state, setState] = useState<'idle' | 'ok' | 'fail'>('idle');
  const onClick = () => {
    copyText(text).then(
      () => setState('ok'),
      () => setState('fail'),
    );
    window.setTimeout(() => setState('idle'), 1500);
  };
  return (
    <button type="button" className="btn small" onClick={onClick} title={`Copy ${label}`}>
      {state === 'ok' ? 'Copied' : state === 'fail' ? 'Failed' : label}
    </button>
  );
}

export function StateBadge({ state }: { state: JobState }) {
  return <span className={`badge state-${state}`}>{state}</span>;
}

export function ErrorBox({ message }: { message: string | null }) {
  if (!message) return null;
  return <div className="error-box">{message}</div>;
}

/** Inline SVG line chart of rate samples, scaled to the max sample. */
export function Sparkline({ values, width = 600, height = 80 }: { values: number[]; width?: number; height?: number }) {
  if (values.length < 2) {
    return (
      <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <text x={width / 2} y={height / 2} textAnchor="middle" dominantBaseline="middle" className="sparkline-empty">
          waiting for samples…
        </text>
      </svg>
    );
  }
  const max = Math.max(...values) || 1;
  const step = width / (values.length - 1);
  const pad = 2;
  const pts = values.map((v, i) => `${(i * step).toFixed(1)},${(height - pad - (v / max) * (height - 2 * pad)).toFixed(1)}`);
  const line = pts.join(' ');
  const area = `0,${height} ${line} ${width},${height}`;
  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <polygon points={area} className="sparkline-area" />
      <polyline points={line} className="sparkline-line" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function BenchTable({ results }: { results: BenchResult[] }) {
  let best = -1;
  results.forEach((r, i) => {
    if (r.verified && (best < 0 || r.rate > results[best].rate)) best = i;
  });
  return (
    <table className="table">
      <thead>
        <tr>
          <th>Backend</th>
          <th>Devices</th>
          <th>Algorithm</th>
          <th className="num">Rate</th>
          <th>Verified</th>
          <th>Error</th>
        </tr>
      </thead>
      <tbody>
        {results.map((r, i) => (
          <tr key={`${r.backend}-${i}`} className={i === best ? 'best' : undefined}>
            <td>
              {r.backend}
              {i === best && <span className="badge best-badge">best</span>}
            </td>
            <td>{r.devices.join(', ') || '—'}</td>
            <td>{r.algorithm}</td>
            <td className="num">{formatRate(r.rate, r.algorithm === 'kangaroo' ? 'jumps/s' : 'keys/s')}</td>
            <td>{r.verified ? <span className="ok">yes</span> : <span className="bad">no</span>}</td>
            <td className="wrap">{r.error || '—'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
