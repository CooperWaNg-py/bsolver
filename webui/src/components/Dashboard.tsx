import { useEffect, useRef, useState } from 'react';
import { api, errorMessage } from '../api';
import { formatDuration, formatPow2, formatRate, formatTime, shortHex } from '../format';
import type { JobStatus, LogEvent } from '../types';
import { BenchTable, CopyButton, ErrorBox, Sparkline, StateBadge } from './common';

type Props = {
  status: JobStatus | null;
  rates: number[];
  logs: LogEvent[];
  onStatus: (s: JobStatus) => void;
};

export function Dashboard({ status, rates, logs, onStatus }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const control = async (action: () => Promise<JobStatus>) => {
    setBusy(true);
    setError(null);
    try {
      onStatus(await action());
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  if (!status) {
    return (
      <section className="panel">
        <p className="muted">Loading job status…</p>
      </section>
    );
  }

  const s = status;
  const canPause = s.state === 'running';
  const canResume = s.state === 'paused';
  const canStop = s.state === 'benchmarking' || s.state === 'running' || s.state === 'paused';
  const eta = s.etaSeconds !== null ? formatDuration(s.etaSeconds) : '—';
  const algorithm = s.algorithm ? (s.order ? `${s.algorithm} (${s.order})` : s.algorithm) : '—';

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>
            {s.id === 0 ? 'No job yet' : `Job #${s.id}`} <StateBadge state={s.state} />
            {s.outcome && <span className={`badge outcome-${s.outcome}`}>{s.outcome}</span>}
            {s.resumed && <span className="badge">resumed</span>}
          </h2>
          <div className="actions">
            <button className="btn" disabled={!canPause || busy} onClick={() => control(api.pauseJob)}>
              Pause
            </button>
            <button className="btn" disabled={!canResume || busy} onClick={() => control(api.resumeJob)}>
              Resume
            </button>
            <button className="btn danger" disabled={!canStop || busy} onClick={() => control(api.stopJob)}>
              Stop
            </button>
          </div>
        </div>
        <ErrorBox message={error} />
        {s.error && <div className="error-box">{s.error}</div>}

        <dl className="stats">
          <div>
            <dt>Puzzle</dt>
            <dd>{s.puzzle !== null ? `#${s.puzzle}` : '—'}</dd>
          </div>
          <div>
            <dt>Algorithm</dt>
            <dd>{algorithm}</dd>
          </div>
          <div>
            <dt>Backend</dt>
            <dd>{s.backend ?? '—'}</dd>
          </div>
          <div>
            <dt>Devices</dt>
            <dd className="wrap">{s.devices.length ? s.devices.join(', ') : '—'}</dd>
          </div>
          <div>
            <dt>Range</dt>
            <dd>{s.rangeBits !== null ? `2^${s.rangeBits.toFixed(2)} keys` : '—'}</dd>
          </div>
          <div>
            <dt>Rate</dt>
            <dd className="big">{formatRate(s.rate, s.rateUnit)}</dd>
          </div>
          <div>
            <dt>Done</dt>
            <dd>
              {formatPow2(s.workDone)}
              {s.workExpected !== null && <span className="muted"> / {formatPow2(s.workExpected)}</span>}
            </dd>
          </div>
          <div>
            <dt>ETA</dt>
            <dd>{eta}</dd>
          </div>
          <div>
            <dt>Elapsed</dt>
            <dd>{formatDuration(s.elapsedSeconds)}</dd>
          </div>
          <div>
            <dt>Started</dt>
            <dd>{formatTime(s.startedAt)}</dd>
          </div>
        </dl>

        {s.rangeStart && s.rangeEnd && (
          <p className="mono range">
            {shortHex(s.rangeStart)} … {shortHex(s.rangeEnd)}
          </p>
        )}

        {s.progress !== null && (
          <div className="progress" title={`${(s.progress * 100).toFixed(4)}%`}>
            <div className="progress-bar" style={{ width: `${Math.min(100, s.progress * 100)}%` }} />
            <span className="progress-label">{(s.progress * 100).toFixed(2)}%</span>
          </div>
        )}

        <div className="chart">
          <div className="chart-head">
            <span>Rate (last {rates.length} samples)</span>
            {rates.length > 0 && <span className="muted">peak {formatRate(Math.max(...rates), s.rateUnit)}</span>}
          </div>
          <Sparkline values={rates} />
        </div>
      </section>

      {s.found.length > 0 && (
        <section className="panel found">
          <h2>Found keys</h2>
          {s.found.map((k) => (
            <div className="found-key" key={`${k.privkey}-${k.address}`}>
              <div className="found-title">
                {k.puzzle !== null ? `Puzzle #${k.puzzle}` : 'Target'} — <span className="mono">{k.address}</span>
              </div>
              <div className="kv">
                <span className="k">Private key</span>
                <span className="mono wrap">{k.privkey}</span>
                <CopyButton text={k.privkey} />
              </div>
              <div className="kv">
                <span className="k">WIF</span>
                <span className="mono wrap">{k.wif}</span>
                <CopyButton text={k.wif} />
              </div>
            </div>
          ))}
        </section>
      )}

      {s.benchmark.length > 0 && (
        <section className="panel">
          <h2>Backend benchmark</h2>
          <BenchTable results={s.benchmark} />
        </section>
      )}

      <LogPanel logs={logs} />
    </>
  );
}

function LogPanel({ logs }: { logs: LogEvent[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [logs]);
  return (
    <section className="panel">
      <h2>Log</h2>
      <div
        className="log"
        ref={ref}
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
        }}
      >
        {logs.length === 0 && <div className="muted">No log messages yet.</div>}
        {logs.map((l, i) => (
          <div key={i} className={`log-line level-${l.level}`}>
            <span className="log-time">{formatTime(l.time)}</span>
            <span className="log-level">{l.level}</span>
            <span className="log-msg">{l.message}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
