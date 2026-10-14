import { useState } from 'react';
import type { Puzzle } from '../types';
import { shortHex } from '../format';
import { ErrorBox } from './common';

type Props = {
  puzzles: Puzzle[] | null;
  error: string | null;
  onStart: (id: number) => void;
};

export function Puzzles({ puzzles, error, onStart }: Props) {
  const [unsolvedOnly, setUnsolvedOnly] = useState(true);
  const rows = (puzzles ?? []).filter((p) => !unsolvedOnly || !p.solved);
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Puzzles</h2>
        <label className="toggle">
          <input type="checkbox" checked={unsolvedOnly} onChange={(e) => setUnsolvedOnly(e.target.checked)} /> Unsolved only
        </label>
      </div>
      <ErrorBox message={error} />
      {puzzles === null && !error && <p className="muted">Loading catalog…</p>}
      {puzzles !== null && (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th className="num">#</th>
                <th className="num">Bits</th>
                <th>Address</th>
                <th>Pubkey</th>
                <th>Status</th>
                <th>Key</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p.id}>
                  <td className="num">{p.id}</td>
                  <td className="num">{p.bits}</td>
                  <td className="mono">{p.address}</td>
                  <td>{p.pubkey ? <span className="ok">yes</span> : <span className="muted">no</span>}</td>
                  <td>{p.solved ? <span className="muted">solved</span> : <span className="ok">unsolved</span>}</td>
                  <td className="mono wrap">{p.privkey ? shortHex(p.privkey) : '—'}</td>
                  <td>
                    <button className="btn small" onClick={() => onStart(p.id)}>
                      Start
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
