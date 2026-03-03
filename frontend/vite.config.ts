import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
  server: {
    port: 5173,
    allowedHosts: ['tonita-deposable-manneristically.ngrok-free.dev'],
    proxy: {
      // Proxy for /api routes
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, ''), // /api/v1 -> /v1
      },
      // Proxy for /portfolio -> /v1/portfolio
      '/portfolio': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/v1' + path, // /portfolio/summary -> /v1/portfolio/summary
      },
      // Proxy for /market -> /v1/market
      '/market': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/v1' + path,
      },
      // Proxy for /orders -> /v1/orders
      '/orders': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/v1' + path,
      },
      // Proxy for /risk -> /v1/risk
      '/risk': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/v1' + path,
      },
      // Proxy for /news -> /v1/news
      '/news': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/v1' + path,
      },
      // Proxy for /auth -> /v1/auth
      '/auth': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/v1' + path,
      },
    },
  },
})
