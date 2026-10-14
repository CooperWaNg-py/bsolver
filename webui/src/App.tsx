import { useCallback, useEffect, useRef, useState } from 'react';
import { api, connectEvents, errorMessage, onAuthNeeded } from './api';
import type { Info, JobStatus, LogEvent, Puzzle } from './types';
import { ACTIVE_STATES } from './types';
import { Dashboard } from './components/Dashboard';
import { NewJob, type Prefill } from './components/NewJob';
import { Puzzles } from './components/Puzzles';
import { Benchmark } from './components/Benchmark';
import { Results } from './components/Results';
import { TokenPrompt } from './components/TokenPrompt';
import { StateBadge } from './components/common';

type Tab = 'dashboard' | 'new' | 'puzzles' | 'bench' | 'results';
type LoadState = 'missing' | 'loading' | 'loaded';

const TAB_LABELS: Record<Tab, string> = {
  dashboard: 'Dashboard',
  new: 'New Job',
  puzzles: 'Puzzles',
  bench: 'Benchmark',
  results: 'Results',
};

const MAX_RATE_SAMPLES = 300;
const MAX_LOG_LINES = 200;

function isTab(id: string): id is Tab {
  return id in TAB_LABELS;
}

function tabFromHash(): Tab {
  const id = location.hash.replace(/^#\/?/, '');
  return isTab(id) ? id : 'dashboard';
}

export function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const [authNeeded, setAuthNeeded] = useState(false);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [rates, setRates] = useState<number[]>([]);
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [info, setInfo] = useState<Info | null>(null);
  const [puzzles, setPuzzles] = useState<Puzzle[] | null>(null);
  const [puzzlesError, setPuzzlesError] = useState<string | null>(null);
  const [prefill, setPrefill] = useState<Prefill | null>(null);
  const [resultsKey, setResultsKey] = useState(0);
  const jobId = useRef<number | null>(null);
  const lastSeq = useRef(0);
  // A new event stream (possibly to a restarted server, whose seq restarts) delivers an authoritative snapshot first.
  const resync = useRef(false);
  const staticLoad = useRef({ info: 'missing' as LoadState, puzzles: 'missing' as LoadState });

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener('hashchange', onHash);
    if (!location.hash) history.replaceState(null, '', '#/dashboard');
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  useEffect(() => onAuthNeeded(setAuthNeeded), []);

  // Snapshots can arrive out of order (an HTTP reply racing the event stream); drop ones older than the shown one.
  // Rate samples belong to one job; a new job id starts a fresh sparkline. Returns whether `s` was applied.
  const applyStatus = useCallback((s: JobStatus, force = false): boolean => {
    if (!force && jobId.current === s.id && s.seq < lastSeq.current) return false;
    if (jobId.current !== s.id) {
      jobId.current = s.id;
      setRates([]);
    }
    lastSeq.current = s.seq;
    setStatus(s);
    return true;
  }, []);

  // /api/info and /api/puzzles are static per server: fetched on mount and again on every (re)connect while missing,
  // so a tab opened while the server was down recovers without a reload.
  const loadStatic = useCallback(() => {
    const st = staticLoad.current;
    if (st.info === 'missing') {
      st.info = 'loading';
      api.info().then(
        (i) => {
          st.info = 'loaded';
          setInfo(i);
        },
        () => {
          st.info = 'missing';
        },
      );
    }
    if (st.puzzles === 'missing') {
      st.puzzles = 'loading';
      api.puzzles().then(
        (p) => {
          st.puzzles = 'loaded';
          setPuzzles(p);
          setPuzzlesError(null);
        },
        (e) => {
          st.puzzles = 'missing';
          setPuzzlesError(errorMessage(e));
        },
      );
    }
  }, []);

  useEffect(() => {
    api.job().then(applyStatus, (e) =>
      setLogs((l) => [...l, { time: new Date().toISOString(), level: 'error', message: `GET /api/job: ${errorMessage(e)}` }]),
    );
    loadStatic();

    return connectEvents({
      status: (s) => {
        const applied = applyStatus(s, resync.current);
        resync.current = false;
        if (applied && ACTIVE_STATES.includes(s.state)) {
          setRates((r) => [...r.slice(-(MAX_RATE_SAMPLES - 1)), s.rate]);
        }
      },
      found: (k) => {
        setResultsKey((n) => n + 1);
        setStatus((s) =>
          s && !s.found.some((f) => f.privkey === k.privkey && f.address === k.address) ? { ...s, found: [...s.found, k] } : s,
        );
      },
      log: (l) => setLogs((prev) => [...prev.slice(-(MAX_LOG_LINES - 1)), l]),
      connection: (c) => {
        setConnected(c);
        if (!c) return;
        resync.current = true;
        loadStatic();
      },
    });
  }, [applyStatus, loadStatic]);

  const onStarted = (s: JobStatus) => {
    applyStatus(s);
    location.hash = '#/dashboard';
  };

  const onStartPuzzle = (id: number) => {
    setPrefill({ puzzle: id, nonce: Date.now() });
    location.hash = '#/new';
  };

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          bsolver{info && <span className="version">v{info.version}</span>}
        </div>
        <nav className="tabs">
          {(Object.keys(TAB_LABELS) as Tab[]).map((t) => (
            <a key={t} href={`#/${t}`} className={t === tab ? 'tab active' : 'tab'}>
              {TAB_LABELS[t]}
            </a>
          ))}
        </nav>
        <div className="header-status">
          {status && status.id !== 0 && <StateBadge state={status.state} />}
          <span className={connected ? 'dot online' : 'dot offline'} title={connected ? 'live' : 'disconnected'} />
        </div>
      </header>

      <main className="main">
        {tab === 'dashboard' && <Dashboard status={status} rates={rates} logs={logs} onStatus={applyStatus} />}
        {/* New Job and Benchmark stay mounted so form values and a running benchmark survive tab switches. */}
        <div hidden={tab !== 'new'}>
          <NewJob info={info} puzzles={puzzles} prefill={prefill} onStarted={onStarted} />
        </div>
        {tab === 'puzzles' && <Puzzles puzzles={puzzles} error={puzzlesError} onStart={onStartPuzzle} />}
        <div hidden={tab !== 'bench'}>
          <Benchmark />
        </div>
        {tab === 'results' && <Results refreshKey={resultsKey} />}
      </main>

      {authNeeded && <TokenPrompt />}
    </div>
  );
}
