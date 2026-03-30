'use client';

import dynamic from 'next/dynamic';

// Mount the React SPA (BrowserRouter-based) entirely on the client side
// ssr: false prevents any server-side rendering since BrowserRouter requires browser APIs
const App = dynamic(() => import('../src/app-src/App'), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center">
        <div className="w-12 h-12 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-text-muted text-sm">Loading AI Trading System...</p>
      </div>
    </div>
  ),
});

export default function ClientApp() {
  return <App />;
}
