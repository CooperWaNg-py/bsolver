import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  // Prefix '' exposes every process environment variable, including BSOLVER_WEBUI_OUT.
  const env = loadEnv(mode, '.', '');
  return {
    base: '/',
    plugins: [react()],
    build: {
      outDir: env.BSOLVER_WEBUI_OUT || 'dist',
      emptyOutDir: true,
    },
    server: {
      proxy: {
        '/api': 'http://127.0.0.1:8080',
      },
    },
  };
});
