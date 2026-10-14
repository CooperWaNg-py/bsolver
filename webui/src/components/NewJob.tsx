import { useEffect, useMemo, useState, type FormEvent } from 'react';
import { api, errorMessage } from '../api';
import type { AddressMode, Algorithm, Info, JobSpec, JobStatus, Order, Puzzle } from '../types';
import { MAX_THREADS } from '../types';
import { ErrorBox } from './common';

export type Prefill = { puzzle: number; nonce: number };

type Props = {
  info: Info | null;
  puzzles: Puzzle[] | null;
  prefill: Prefill | null;
  onStarted: (s: JobStatus) => void;
};

type TargetMode = 'puzzle' | 'custom';
type RangeMode = 'default' | 'range' | 'bits';
type AlgoChoice = 'auto' | Algorithm;

const HEX_RE = /^(0x)?[0-9a-fA-F]{1,64}$/;

/** Parses an optional non-negative integer field; '' means "not set". */
function parseOptInt(label: string, value: string, min: number, max: number): number | undefined {
  const t = value.trim();
  if (t === '') return undefined;
  const n = Number(t);
  if (!Number.isInteger(n) || n < min || n > max) throw new Error(`${label} must be an integer ${min}-${max}`);
  return n;
}

export function NewJob({ info, puzzles, prefill, onStarted }: Props) {
  const [targetMode, setTargetMode] = useState<TargetMode>('puzzle');
  const [puzzleId, setPuzzleId] = useState<number | null>(null);
  const [addresses, setAddresses] = useState('');
  const [pubkey, setPubkey] = useState('');
  const [rangeMode, setRangeMode] = useState<RangeMode>('default');
  const [rangeStart, setRangeStart] = useState('');
  const [rangeEnd, setRangeEnd] = useState('');
  const [bits, setBits] = useState('');
  const [algorithm, setAlgorithm] = useState<AlgoChoice>('auto');
  const [order, setOrder] = useState<Order>('sequential');
  const [backend, setBackend] = useState('auto');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [threads, setThreads] = useState('');
  const [devices, setDevices] = useState('');
  const [bsgsMemoryMb, setBsgsMemoryMb] = useState('');
  const [dpBits, setDpBits] = useState('');
  const [addressMode, setAddressMode] = useState<AddressMode>('compressed');
  const [checkpoint, setCheckpoint] = useState(true);
  const [fresh, setFresh] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Unsolved puzzles first, each group by id.
  const sortedPuzzles = useMemo(
    () => (puzzles ? [...puzzles].sort((a, b) => Number(a.solved) - Number(b.solved) || a.id - b.id) : []),
    [puzzles],
  );

  // Default selection: the first unsolved puzzle.
  useEffect(() => {
    if (puzzleId === null && sortedPuzzles.length > 0) setPuzzleId(sortedPuzzles[0].id);
  }, [sortedPuzzles, puzzleId]);

  useEffect(() => {
    if (!prefill) return;
    setTargetMode('puzzle');
    setPuzzleId(prefill.puzzle);
    setRangeMode('default');
    setError(null);
  }, [prefill]);

  const puzzle = sortedPuzzles.find((p) => p.id === puzzleId) ?? null;
  const hasPubkey = targetMode === 'puzzle' ? puzzle?.pubkey != null : pubkey.trim() !== '';
  const effectiveAlgo: AlgoChoice = !hasPubkey && (algorithm === 'bsgs' || algorithm === 'kangaroo') ? 'auto' : algorithm;
  const isScan = effectiveAlgo === 'scan' || (effectiveAlgo === 'auto' && !hasPubkey);
  const isBsgs = effectiveAlgo === 'bsgs';
  const isKangaroo = effectiveAlgo === 'kangaroo' || (effectiveAlgo === 'auto' && hasPubkey);
  const effectiveRangeMode: RangeMode = targetMode === 'custom' && rangeMode === 'default' ? 'range' : rangeMode;

  useEffect(() => {
    if (effectiveAlgo !== algorithm) setAlgorithm(effectiveAlgo);
  }, [effectiveAlgo, algorithm]);

  const backends = (info?.backends ?? []).filter((b) => b.available);
  // With auto algorithm the concrete one is decided server-side; offer every available backend.
  const backendOptions = effectiveAlgo === 'auto' ? backends : backends.filter((b) => b.algorithms.includes(effectiveAlgo));
  useEffect(() => {
    if (backend !== 'auto' && info && !backendOptions.some((b) => b.id === backend)) setBackend('auto');
  }, [backend, backendOptions, info]);

  const buildSpec = (): JobSpec => {
    const spec: JobSpec = {};
    if (targetMode === 'puzzle') {
      if (puzzleId === null) throw new Error('select a puzzle');
      spec.puzzle = puzzleId;
    } else {
      const addrs = addresses
        .split(/[\s,]+/)
        .map((a) => a.trim())
        .filter(Boolean);
      if (addrs.length) spec.addresses = addrs;
      const pk = pubkey.trim();
      if (pk) {
        if (!/^(0x)?[0-9a-fA-F]+$/.test(pk) || ![66, 130].includes(pk.replace(/^0x/, '').length)) {
          throw new Error('public key must be 33 or 65 bytes of hex');
        }
        spec.pubkey = pk;
      }
      if (!addrs.length && !pk) throw new Error('give at least one address or a public key');
    }

    if (effectiveRangeMode === 'range') {
      const a = rangeStart.trim();
      const b = rangeEnd.trim();
      if (!HEX_RE.test(a) || !HEX_RE.test(b)) throw new Error('range start and end must be hex (up to 64 digits)');
      spec.rangeStart = a;
      spec.rangeEnd = b;
    } else if (effectiveRangeMode === 'bits') {
      const n = parseOptInt('bits', bits, 1, 256);
      if (n === undefined) throw new Error('bits is required');
      spec.bits = n;
    }

    if (effectiveAlgo !== 'auto') spec.algorithm = effectiveAlgo;
    if (isScan && order !== 'sequential') spec.order = order;
    if (isScan && addressMode !== 'compressed') spec.addressMode = addressMode;
    if (backend !== 'auto') spec.backend = backend;

    const t = parseOptInt('threads', threads, 0, MAX_THREADS);
    if (t !== undefined) spec.threads = t;
    const devs = devices.trim();
    if (devs) {
      spec.devices = devs.split(',').map((d) => {
        const n = Number(d.trim());
        if (d.trim() === '' || !Number.isInteger(n) || n < 0) throw new Error(`invalid device index '${d.trim()}'`);
        return n;
      });
    }
    if (isBsgs) {
      const m = parseOptInt('BSGS memory (MB)', bsgsMemoryMb, 1, 1 << 30);
      if (m !== undefined) spec.bsgsMemoryMb = m;
    }
    if (isKangaroo) {
      const dp = parseOptInt('DP bits', dpBits, 0, 60);
      if (dp !== undefined) spec.dpBits = dp;
    }
    if (!checkpoint) spec.checkpoint = false;
    if (fresh) spec.fresh = true;
    return spec;
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    let spec: JobSpec;
    try {
      spec = buildSpec();
    } catch (err) {
      setError(errorMessage(err));
      return;
    }
    setSubmitting(true);
    try {
      onStarted(await api.startJob(spec));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="panel form" onSubmit={onSubmit}>
      <h2>New job</h2>

      <fieldset>
        <legend>Target</legend>
        <div className="radio-row">
          <label>
            <input type="radio" checked={targetMode === 'puzzle'} onChange={() => setTargetMode('puzzle')} /> Puzzle
          </label>
          <label>
            <input type="radio" checked={targetMode === 'custom'} onChange={() => setTargetMode('custom')} /> Custom
            targets
          </label>
        </div>

        {targetMode === 'puzzle' ? (
          <div className="field">
            <label htmlFor="puzzle">Puzzle</label>
            {puzzles === null ? (
              <span className="muted">Loading catalog…</span>
            ) : (
              <select id="puzzle" value={puzzleId ?? ''} onChange={(e) => setPuzzleId(Number(e.target.value))}>
                {sortedPuzzles.map((p) => (
                  <option key={p.id} value={p.id}>
                    #{p.id} — {p.bits} bits — pubkey {p.pubkey ? 'known' : 'unknown'}
                    {p.solved ? ' — solved' : ''}
                  </option>
                ))}
              </select>
            )}
            {puzzle && (
              <div className="hint mono">
                {puzzle.address} · range {puzzle.rangeStart}…{puzzle.rangeEnd}
              </div>
            )}
          </div>
        ) : (
          <>
            <div className="field">
              <label htmlFor="addresses">Addresses (P2PKH, one per line)</label>
              <textarea
                id="addresses"
                rows={4}
                spellCheck={false}
                value={addresses}
                onChange={(e) => setAddresses(e.target.value)}
                placeholder="1..."
              />
            </div>
            <div className="field">
              <label htmlFor="pubkey">Public key (hex, optional)</label>
              <input
                id="pubkey"
                className="mono"
                spellCheck={false}
                value={pubkey}
                onChange={(e) => setPubkey(e.target.value)}
                placeholder="02… / 03… / 04…"
              />
            </div>
          </>
        )}

        <div className="field">
          <label>Range</label>
          <div className="radio-row">
            {targetMode === 'puzzle' && (
              <label>
                <input type="radio" checked={effectiveRangeMode === 'default'} onChange={() => setRangeMode('default')} />{' '}
                Puzzle range
              </label>
            )}
            <label>
              <input type="radio" checked={effectiveRangeMode === 'range'} onChange={() => setRangeMode('range')} /> Start
              : end (hex)
            </label>
            <label>
              <input type="radio" checked={effectiveRangeMode === 'bits'} onChange={() => setRangeMode('bits')} /> Bits
            </label>
          </div>
          {effectiveRangeMode === 'range' && (
            <div className="inline">
              <input
                className="mono"
                spellCheck={false}
                placeholder="start hex"
                value={rangeStart}
                onChange={(e) => setRangeStart(e.target.value)}
              />
              <span>:</span>
              <input
                className="mono"
                spellCheck={false}
                placeholder="end hex"
                value={rangeEnd}
                onChange={(e) => setRangeEnd(e.target.value)}
              />
            </div>
          )}
          {effectiveRangeMode === 'bits' && (
            <input type="number" min={1} max={256} placeholder="1-256" value={bits} onChange={(e) => setBits(e.target.value)} />
          )}
        </div>
      </fieldset>

      <fieldset>
        <legend>Search</legend>
        <div className="grid2">
          <div className="field">
            <label htmlFor="algorithm">Algorithm</label>
            <select id="algorithm" value={effectiveAlgo} onChange={(e) => setAlgorithm(e.target.value as AlgoChoice)}>
              <option value="auto">auto</option>
              <option value="scan">scan</option>
              <option value="bsgs" disabled={!hasPubkey}>
                bsgs{hasPubkey ? '' : ' (needs public key)'}
              </option>
              <option value="kangaroo" disabled={!hasPubkey}>
                kangaroo{hasPubkey ? '' : ' (needs public key)'}
              </option>
            </select>
          </div>
          {isScan && (
            <div className="field">
              <label htmlFor="order">Order</label>
              <select id="order" value={order} onChange={(e) => setOrder(e.target.value as Order)}>
                <option value="sequential">sequential</option>
                <option value="random">random</option>
              </select>
            </div>
          )}
          <div className="field">
            <label htmlFor="backend">Backend</label>
            <select id="backend" value={backend} onChange={(e) => setBackend(e.target.value)}>
              <option value="auto">auto (benchmark)</option>
              {backendOptions.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.id}
                  {b.devices.length ? ` — ${b.devices.map((d) => d.name).join(', ')}` : ''}
                </option>
              ))}
            </select>
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>
          <button type="button" className="link" onClick={() => setShowAdvanced(!showAdvanced)}>
            {showAdvanced ? '▾' : '▸'} Advanced
          </button>
        </legend>
        {showAdvanced && (
          <div className="grid2">
            <div className="field">
              <label htmlFor="threads">Threads (blank/0 = all cores)</label>
              <input
                id="threads"
                type="number"
                min={0}
                max={MAX_THREADS}
                value={threads}
                onChange={(e) => setThreads(e.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="devices">Devices (comma list, blank = all)</label>
              <input id="devices" placeholder="0,1" value={devices} onChange={(e) => setDevices(e.target.value)} />
            </div>
            {isBsgs && (
              <div className="field">
                <label htmlFor="bsgsmem">BSGS memory (MB, default 1024)</label>
                <input
                  id="bsgsmem"
                  type="number"
                  min={1}
                  value={bsgsMemoryMb}
                  onChange={(e) => setBsgsMemoryMb(e.target.value)}
                />
              </div>
            )}
            {isKangaroo && (
              <div className="field">
                <label htmlFor="dpbits">DP bits (blank = automatic)</label>
                <input id="dpbits" type="number" min={0} max={60} value={dpBits} onChange={(e) => setDpBits(e.target.value)} />
              </div>
            )}
            {isScan && (
              <div className="field">
                <label htmlFor="addrmode">Address mode</label>
                <select id="addrmode" value={addressMode} onChange={(e) => setAddressMode(e.target.value as AddressMode)}>
                  <option value="compressed">compressed</option>
                  <option value="uncompressed">uncompressed</option>
                  <option value="both">both</option>
                </select>
              </div>
            )}
            <div className="field checks">
              <label>
                <input type="checkbox" checked={checkpoint} onChange={(e) => setCheckpoint(e.target.checked)} /> Checkpoint
                (resume automatically)
              </label>
              <label>
                <input type="checkbox" checked={fresh} onChange={(e) => setFresh(e.target.checked)} /> Fresh start (ignore
                existing checkpoint)
              </label>
            </div>
          </div>
        )}
      </fieldset>

      <ErrorBox message={error} />
      <div className="actions">
        <button type="submit" className="btn primary" disabled={submitting}>
          {submitting ? 'Starting…' : 'Start job'}
        </button>
      </div>
    </form>
  );
}
