import { useEffect, useState, type FormEvent } from 'react';
import { api, errorMessage } from '../api';
import type { BenchResult } from '../types';
import { MAX_BENCH_SECONDS } from '../types';
import { BenchTable, ErrorBox } from './common';

export function Benchmark() {
  const [algorithm, setAlgorithm] = useState<'scan' | 'kangaroo'>('scan');
  const [seconds, setSeconds] = useState('2');
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<BenchResult[] | null>(null);

  useEffect(() => {
    if (!running) return;
    const started = Date.now();
    setElapsed(0);
    const id = window.setInterval(() => setElapsed((Date.now() - started) / 1000), 250);
    return () => clearInterval(id);
  }, [running]);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const secs = Number(seconds);
    if (!Number.isFinite(secs) || secs <= 0 || secs > MAX_BENCH_SECONDS) {
      setError(`seconds must be a number in (0, ${MAX_BENCH_SECONDS}]`);
      return;
    }
    setError(null);
    setRunning(true);
    try {
      setResults(await api.bench({ algorithm, seconds: secs }));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="panel">
      <h2>Benchmark</h2>
      <p className="muted">
        Runs the correctness check and a throughput test on every available backend. Not possible while a job is
        active.
      </p>
      <form className="inline" onSubmit={onSubmit}>
        <label>
          Algorithm{' '}
          <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value === 'kangaroo' ? 'kangaroo' : 'scan')}>
            <option value="scan">scan</option>
            <option value="kangaroo">kangaroo</option>
          </select>
        </label>
        <label>
          Seconds per backend{' '}
          <input
            type="number"
            min={0.5}
            max={MAX_BENCH_SECONDS}
            step={0.5}
            value={seconds}
            onChange={(e) => setSeconds(e.target.value)}
          />
        </label>
        <button type="submit" className="btn primary" disabled={running}>
          {running ? `Running… ${elapsed.toFixed(0)}s` : 'Run benchmark'}
        </button>
      </form>
      <ErrorBox message={error} />
      {results !== null &&
        (results.length ? <BenchTable results={results} /> : <p className="muted">No backend supports this algorithm.</p>)}
    </section>
  );
}
