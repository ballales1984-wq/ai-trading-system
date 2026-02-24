import { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';

const Marketing = lazy(() => import('./pages/Marketing'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Portfolio = lazy(() => import('./pages/Portfolio'));
const Market = lazy(() => import('./pages/Market'));
const Orders = lazy(() => import('./pages/Orders'));
const Terms = lazy(() => import('./pages/Terms'));
const Privacy = lazy(() => import('./pages/Privacy'));
const RiskDisclosure = lazy(() => import('./pages/RiskDisclosure'));

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
        <Route
          path="/"
          element={
            <Suspense fallback={<PageFallback />}>
              <Marketing />
            </Suspense>
          }
        />
        <Route path="/" element={<Layout />}>
          <Route path="dashboard" element={<Suspense fallback={<PageFallback />}><Dashboard /></Suspense>} />
          <Route path="portfolio" element={<Suspense fallback={<PageFallback />}><Portfolio /></Suspense>} />
          <Route path="market" element={<Suspense fallback={<PageFallback />}><Market /></Suspense>} />
          <Route path="orders" element={<Suspense fallback={<PageFallback />}><Orders /></Suspense>} />
          <Route path="legal/terms" element={<Suspense fallback={<PageFallback />}><Terms /></Suspense>} />
          <Route path="legal/privacy" element={<Suspense fallback={<PageFallback />}><Privacy /></Suspense>} />
          <Route path="legal/risk" element={<Suspense fallback={<PageFallback />}><RiskDisclosure /></Suspense>} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;

