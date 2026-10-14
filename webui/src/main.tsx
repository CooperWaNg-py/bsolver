import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { consumeHashToken } from './api';
import { App } from './App';
import './styles.css';

// The `#token=` fragment must be consumed before the App reads location.hash for tab routing.
consumeHashToken();

const root = document.getElementById('root');
if (!root) throw new Error('missing #root element');
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
