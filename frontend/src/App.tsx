import { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import Portfolio from './pages/Portfolio';
import Market from './pages/Market';
import Orders from './pages/Orders';
import News from './pages/News';
import Marketing from './pages/Marketing';
import Login from './pages/Login';
import Strategy from './pages/Strategy';
import Risk from './pages/Risk';
import Settings from './pages/Settings';
import MLMonitoring from './pages/MLMonitoring';
import InvestorPortal from './pages/InvestorPortal';
import AIAssistant from './pages/AIAssistant';

// Lazy load payment page only
const PaymentTest = lazy(() => import('./pages/PaymentTest'));

function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        {/* Public routes */}
        <Route path="/" element={<Marketing />} />
        <Route path="/marketing" element={<Marketing />} />
        <Route path="/login" element={<Login />} />
        <Route path="/payment" element={<Suspense fallback={<div>Loading...</div>}><PaymentTest /></Suspense>} />
        
        {/* Protected routes - Layout already handles nested Outlets */}
        <Route element={<Layout />}>
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="portfolio" element={<Portfolio />} />
          <Route path="market" element={<Market />} />
          <Route path="orders" element={<Orders />} />
          <Route path="news" element={<News />} />
          <Route path="strategy" element={<Strategy />} />
          <Route path="risk" element={<Risk />} />
          <Route path="settings" element={<Settings />} />
          
          {/* New unified pages - replacing 8050, 8051, 8502 */}
          <Route path="ml-monitoring" element={<MLMonitoring />} />
          <Route path="investor-portal" element={<InvestorPortal />} />
          <Route path="ai-assistant" element={<AIAssistant />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

