import { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Analytics } from '@vercel/analytics/react';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import Portfolio from './pages/Portfolio';
import Market from './pages/Market';
import Orders from './pages/Orders';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="portfolio" element={<Portfolio />} />
          <Route path="market" element={<Market />} />
          <Route path="orders" element={<Orders />} />
        </Route>
      </Routes>
      <Analytics />
    </BrowserRouter>
  );
}

export default App;

