import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api';
import { formatTime } from '../format';
import type { FoundKey } from '../types';
import { CopyButton, ErrorBox } from './common';

/** Table of `/api/results`; reloads whenever `refreshKey` changes (e.g. on a `found` event). */
export function Results({ refreshKey }: { refreshKey: number }) {
  const [results, setResults] = useState<FoundKey[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let cancelled = false;
    api.results().then(
      (r) => {
        if (cancelled) return;
        setResults(r);
        setError(null);
      },
      (e) => {
        if (!cancelled) setError(errorMessage(e));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [refreshKey, reload]);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Results</h2>
        <button className="btn" onClick={() => setReload(reload + 1)}>
          Refresh
        </button>
      </div>
      <ErrorBox message={error} />
      {results === null && !error && <p className="muted">Loading…</p>}
      {results !== null && results.length === 0 && <p className="muted">No keys found yet.</p>}
      {results !== null && results.length > 0 && (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Time</th>
                <th className="num">Puzzle</th>
                <th>Address</th>
                <th>Private key</th>
                <th>WIF</th>
                <th>Algorithm</th>
                <th>Backend</th>
              </tr>
            </thead>
            <tbody>
              {[...results].reverse().map((k, i) => (
                <tr key={`${k.time}-${k.privkey}-${i}`}>
                  <td>{formatTime(k.time)}</td>
                  <td className="num">{k.puzzle ?? '—'}</td>
                  <td className="mono">{k.address}</td>
                  <td className="mono wrap">
                    {k.privkey} <CopyButton text={k.privkey} />
                  </td>
                  <td className="mono wrap">
                    {k.wif} <CopyButton text={k.wif} />
                  </td>
                  <td>{k.algorithm}</td>
                  <td>{k.backend}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
