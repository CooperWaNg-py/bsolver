// Wire types for the bsolver HTTP API. Keep in sync with the server's JSON.

export type Algorithm = 'scan' | 'bsgs' | 'kangaroo';
export type Order = 'sequential' | 'random';
export type AddressMode = 'compressed' | 'uncompressed' | 'both';

export type JobSpec = {
  puzzle?: number | null;
  addresses?: string[];
  pubkey?: string | null;
  rangeStart?: string | null;
  rangeEnd?: string | null;
  bits?: number | null;
  algorithm?: 'auto' | Algorithm;
  order?: Order;
  addressMode?: AddressMode;
  backend?: string;
  threads?: number;
  devices?: number[];
  bsgsMemoryMb?: number;
  dpBits?: number | null;
  checkpoint?: boolean;
  fresh?: boolean;
  stopOnFound?: boolean;
};

export type FoundKey = {
  time: string;
  puzzle: number | null;
  address: string;
  pubkey: string;
  privkey: string;
  wif: string;
  algorithm: string;
  backend: string;
};

export type BenchResult = {
  backend: string;
  devices: string[];
  algorithm: Algorithm;
  rate: number;
  verified: boolean;
  error: string;
};

export type JobState = 'idle' | 'benchmarking' | 'running' | 'paused' | 'stopping' | 'finished' | 'failed';
export type JobOutcome = 'found' | 'exhausted' | 'stopped' | 'error';
export type RateUnit = 'keys/s' | 'jumps/s';

export type JobStatus = {
  id: number;
  /** Snapshot sequence number; of two snapshots of the same job, the higher seq is newer. */
  seq: number;
  state: JobState;
  outcome: JobOutcome | null;
  error: string | null;
  spec: JobSpec | null;
  puzzle: number | null;
  algorithm: Algorithm | null;
  order: Order | null;
  backend: string | null;
  devices: string[];
  rangeStart: string | null;
  rangeEnd: string | null;
  rangeBits: number | null;
  rate: number;
  rateUnit: RateUnit;
  workDone: number;
  workExpected: number | null;
  progress: number | null;
  etaSeconds: number | null;
  elapsedSeconds: number;
  startedAt: string | null;
  resumed: boolean;
  exhausted: boolean; // the whole range was covered (scan/bsgs)
  found: FoundKey[];
  benchmark: BenchResult[];
};

export type DeviceInfo = { index: number; name: string; memoryBytes: number; computeUnits: number };

export type BackendInfo = {
  id: string;
  available: boolean;
  reason: string;
  devices: DeviceInfo[];
  algorithms: Algorithm[];
};

export type Info = { version: string; backends: BackendInfo[] };

export type Puzzle = {
  id: number;
  address: string;
  hash160: string;
  pubkey: string | null;
  privkey: string | null;
  solved: boolean;
  bits: number;
  rangeStart: string;
  rangeEnd: string;
};

export type LogLevel = 'info' | 'warn' | 'error';
export type LogEvent = { time: string; level: LogLevel; message: string };

export type BenchRequest = { algorithm: 'scan' | 'kangaroo'; seconds: number };

export const ACTIVE_STATES: readonly JobState[] = ['benchmarking', 'running', 'paused', 'stopping'];

/** Server-side limits (kMaxThreads in job_spec.h, the /api/bench seconds check in http_server.cpp). */
export const MAX_THREADS = 1024;
export const MAX_BENCH_SECONDS = 60;
