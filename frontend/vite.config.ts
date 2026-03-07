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
      // Proxy for /api routes to /api/v1
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/api/v1' + path.replace(/^\/api/, ''), // /api/v1 -> /api/v1
      },
      // Proxy for /portfolio -> /api/v1/portfolio
      '/portfolio': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/api/v1' + path, // /portfolio/summary -> /api/v1/portfolio/summary
      },
      // Proxy for /market -> /api/v1/market
      '/market': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/api/v1' + path,
      },
      // Proxy for /orders -> /api/v1/orders
      '/orders': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/api/v1' + path,
      },
      // Proxy for /risk -> /api/v1/risk
      '/risk': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/api/v1' + path,
      },
      // Proxy for /news -> /api/v1/news
      '/news': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/api/v1' + path,
      },
      // Proxy for /auth -> /api/v1/auth
      '/auth': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => '/api/v1' + path,
      },
    },
  },
})

