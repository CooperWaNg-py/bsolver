import type { BenchRequest, BenchResult, FoundKey, Info, JobSpec, JobStatus, LogEvent, Puzzle } from './types';

const TOKEN_KEY = 'bsolver.token';

/** Error from a failed API call; `message` is the server's `{"error": ...}` text when present. */
export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// ---------------------------------------------------------------------------
// Token handling

/**
 * Moves a `#token=<t>` URL fragment into localStorage and strips it from the address bar.
 * Must run before hash-based tab routing reads `location.hash`.
 */
export function consumeHashToken(): void {
  const hash = location.hash;
  if (!hash.startsWith('#token=')) return;
  const raw = hash.slice('#token='.length);
  let token: string;
  try {
    token = decodeURIComponent(raw);
  } catch {
    token = raw; // malformed escape (e.g. a literal '%'): use the fragment as typed
  }
  if (token) setStoredToken(token);
  history.replaceState(null, '', location.pathname + location.search);
}

function setStoredToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    memoryToken = token;
  }
}

let memoryToken = '';

export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? memoryToken;
  } catch {
    return memoryToken;
  }
}

type AuthListener = (needed: boolean) => void;
const authListeners = new Set<AuthListener>();
const tokenListeners = new Set<() => void>();
let authNeeded = false;
let tokenWaiters: Array<() => void> = [];

function setAuthNeeded(value: boolean): void {
  if (authNeeded === value) return;
  authNeeded = value;
  authListeners.forEach((l) => l(value));
}

/** Subscribes to "token prompt required" changes; returns an unsubscribe function. */
export function onAuthNeeded(listener: AuthListener): () => void {
  authListeners.add(listener);
  listener(authNeeded);
  return () => {
    authListeners.delete(listener);
  };
}

/** Stores a new token, closes the prompt and retries every request that failed with 401. */
export function setToken(token: string): void {
  setStoredToken(token.trim());
  setAuthNeeded(false);
  const waiters = tokenWaiters;
  tokenWaiters = [];
  waiters.forEach((w) => w());
  tokenListeners.forEach((l) => l());
}

function waitForToken(): Promise<void> {
  setAuthNeeded(true);
  return new Promise<void>((resolve) => tokenWaiters.push(resolve));
}

// ---------------------------------------------------------------------------
// Fetch wrapper

async function request<T>(method: 'GET' | 'POST', path: string, body?: unknown): Promise<T> {
  for (;;) {
    const headers: Record<string, string> = { Accept: 'application/json' };
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
    if (body !== undefined) headers['Content-Type'] = 'application/json';

    const res = await fetch(path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });

    if (res.status === 401) {
      await waitForToken();
      continue;
    }

    const text = await res.text();
    let data: unknown = undefined;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        if (res.ok) throw new ApiError(res.status, `invalid JSON response from ${path}`);
      }
    }
    if (!res.ok) {
      let msg = text || `${res.status} ${res.statusText}`;
      if (data && typeof data === 'object' && 'error' in data && typeof data.error === 'string') msg = data.error;
      throw new ApiError(res.status, msg);
    }
    return data as T;
  }
}

export const api = {
  info: () => request<Info>('GET', '/api/info'),
  puzzles: () => request<Puzzle[]>('GET', '/api/puzzles'),
  job: () => request<JobStatus>('GET', '/api/job'),
  startJob: (spec: JobSpec) => request<JobStatus>('POST', '/api/job', spec),
  stopJob: () => request<JobStatus>('POST', '/api/job/stop'),
  pauseJob: () => request<JobStatus>('POST', '/api/job/pause'),
  resumeJob: () => request<JobStatus>('POST', '/api/job/resume'),
  bench: (req: BenchRequest) => request<BenchResult[]>('POST', '/api/bench', req),
  results: () => request<FoundKey[]>('GET', '/api/results'),
};

/** Formats any thrown value for inline display. */
export function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

// ---------------------------------------------------------------------------
// Server-sent events

export type EventHandlers = {
  status: (s: JobStatus) => void;
  found: (k: FoundKey) => void;
  log: (l: LogEvent) => void;
  connection: (connected: boolean) => void;
};

/**
 * Opens `/api/events` and keeps it open: on error the stream is closed and recreated with
 * exponential backoff (1 s .. 30 s). EventSource hides HTTP status codes, so after an error an
 * authenticated probe request runs first; a 401 there opens the token prompt and the stream
 * reconnects as soon as a token is entered. Returns a function that closes the stream for good.
 */
export function connectEvents(handlers: EventHandlers): () => void {
  let source: EventSource | null = null;
  let timer: number | undefined;
  let delay = 1000;
  let closed = false;
  let generation = 0;

  const parse = <T,>(ev: Event, fn: (v: T) => void) => {
    if (!(ev instanceof MessageEvent) || typeof ev.data !== 'string') return;
    try {
      fn(JSON.parse(ev.data) as T);
    } catch {
      // Malformed event payload; ignore it.
    }
  };

  const open = () => {
    if (closed) return;
    timer = undefined;
    const gen = ++generation;
    const es = new EventSource(`/api/events?token=${encodeURIComponent(getToken())}`);
    source = es;
    es.onopen = () => {
      delay = 1000;
      handlers.connection(true);
    };
    es.addEventListener('status', (ev) => parse<JobStatus>(ev, handlers.status));
    es.addEventListener('found', (ev) => parse<FoundKey>(ev, handlers.found));
    es.addEventListener('log', (ev) => parse<LogEvent>(ev, handlers.log));
    es.onerror = () => {
      if (gen !== generation) return;
      es.close();
      source = null;
      handlers.connection(false);
      const wait = delay;
      delay = Math.min(delay * 2, 30000);
      // The probe resolves once authenticated (waiting for the prompt if needed) or fails on
      // network errors; either way reconnect after the backoff delay.
      api
        .info()
        .catch(() => undefined)
        .finally(() => {
          if (closed || gen !== generation) return;
          timer = window.setTimeout(open, wait);
        });
    };
  };

  const reconnectNow = () => {
    if (closed) return;
    generation++;
    clearTimeout(timer);
    source?.close();
    source = null;
    delay = 1000;
    open();
  };

  tokenListeners.add(reconnectNow);
  open();

  return () => {
    closed = true;
    generation++;
    tokenListeners.delete(reconnectNow);
    clearTimeout(timer);
    source?.close();
    source = null;
  };
}
