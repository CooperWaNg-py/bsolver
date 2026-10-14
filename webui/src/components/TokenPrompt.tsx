import { useState, type FormEvent } from 'react';
import { getToken, setToken } from '../api';

/** Modal asking for the API bearer token; submitting retries every request that got a 401. */
export function TokenPrompt() {
  const [value, setValue] = useState(getToken());
  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (value.trim()) setToken(value);
  };
  return (
    <div className="modal-backdrop">
      <form className="modal" onSubmit={onSubmit}>
        <h2>Access token required</h2>
        <p className="muted">
          The bsolver server rejected the current token. Enter the token printed at startup (the <code>#token=</code>{' '}
          part of the Web UI URL).
        </p>
        <input
          type="password"
          autoFocus
          autoComplete="current-password"
          placeholder="token"
          value={value}
          onChange={(e) => setValue(e.target.value)}
        />
        <div className="actions">
          <button type="submit" className="btn primary" disabled={!value.trim()}>
            Save &amp; retry
          </button>
        </div>
      </form>
    </div>
  );
}
