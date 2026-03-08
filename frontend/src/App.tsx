import { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import Portfolio from './pages/Portfolio';
import Market from './pages/Market';
import Orders from './pages/Orders';

// Lazy load login and payment pages
const Login = lazy(() => import('./pages/Login'));
const PaymentTest = lazy(() => import('./pages/PaymentTest'));

function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        {/* Public routes - without Layout */}
        <Route path="/login" element={<Suspense fallback={<div>Loading...</div>}><Login /></Suspense>} />
        <Route path="/payment" element={<Suspense fallback={<div>Loading...</div>}><PaymentTest /></Suspense>} />
        
        {/* Protected routes - with Layout */}
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="portfolio" element={<Portfolio />} />
          <Route path="market" element={<Market />} />
          <Route path="orders" element={<Orders />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

