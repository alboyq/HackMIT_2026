import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

const hub = process.env.HUB_ORIGIN || 'http://localhost:8000'
const hubWs = hub.replace(/^http/, 'ws')

export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
  server: {
    proxy: {
      '/ws': { target: hubWs, ws: true },
      '/api': hub,
      '/snapshots': hub,
    },
  },
})
