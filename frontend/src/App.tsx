import { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Portfolio = lazy(() => import('./pages/Portfolio'));
const Market = lazy(() => import('./pages/Market'));
const Orders = lazy(() => import('./pages/Orders'));

function PageFallback() {
  return (
    <div className="p-6">
      <div className="bg-surface border border-border rounded-lg p-4 text-text-muted">
        Loading page...
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Suspense fallback={<PageFallback />}><Dashboard /></Suspense>} />
          <Route path="portfolio" element={<Suspense fallback={<PageFallback />}><Portfolio /></Suspense>} />
          <Route path="market" element={<Suspense fallback={<PageFallback />}><Market /></Suspense>} />
          <Route path="orders" element={<Suspense fallback={<PageFallback />}><Orders /></Suspense>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

