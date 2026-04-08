import { useState, useEffect } from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { LayoutDashboard, PieChart, TrendingUp, ClipboardList, Menu, X, Bot, FileText, Target, Shield, Settings, AlertTriangle, Brain, Users } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { marketApi } from '../../services/api';

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/portfolio', icon: PieChart, label: 'Portfolio' },
  { to: '/market', icon: TrendingUp, label: 'Market' },
  { to: '/orders', icon: ClipboardList, label: 'Orders' },
  { to: '/news', icon: FileText, label: 'News' },
  { to: '/strategy', icon: Target, label: 'Strategy' },
  { to: '/risk', icon: Shield, label: 'Risk' },
  { to: '/ml-monitoring', icon: Brain, label: 'ML Monitoring' },
  { to: '/investor-portal', icon: Users, label: 'Investor Portal' },
  { to: '/ai-assistant', icon: Bot, label: 'AI Assistant' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const location = useLocation();

  const { data: prices } = useQuery({
    queryKey: ['market-prices'],
    queryFn: marketApi.getAllPrices,
    retry: 1,
    staleTime: 60000,
  });

  const isUsingFallback = !prices?.markets || prices.markets.length === 0;

  // Detect mobile on mount and resize
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (!mobile) setSidebarOpen(false);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Close sidebar on route change
  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  // Close on escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && sidebarOpen) setSidebarOpen(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [sidebarOpen]);

  return (
    <div className="flex min-h-screen bg-background">
      {/* Mobile hamburger button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className={`fixed top-4 left-4 z-50 p-2 rounded-md bg-surface border border-border hover:bg-surface-hover ${isMobile ? 'block' : 'hidden'}`}
        aria-label={sidebarOpen ? 'Close menu' : 'Open menu'}
      >
        {sidebarOpen ? <X size={24} color="#c9d1d9" /> : <Menu size={24} color="#c9d1d9" />}
      </button>

      {/* Sidebar - always hidden on mobile, shown with transform */}
      <aside className={`
        fixed left-0 top-0 h-full w-64 z-40 flex flex-col
        bg-bg-primary border-r border-border
        transition-transform duration-300 ease-in-out
        ${isMobile 
          ? (sidebarOpen ? 'translate-x-0' : '-translate-x-full') 
          : 'translate-x-0'}
      `}>
        {/* Mobile close button inside sidebar */}
        {isMobile && (
          <button
            onClick={() => setSidebarOpen(false)}
            className="absolute top-4 right-4 p-1 rounded hover:bg-surface-hover"
          >
            <X size={24} color="#c9d1d9" />
          </button>
        )}

        {/* Logo */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-tr from-primary to-purple flex items-center justify-center">
              <Bot size={20} color="#fff" />
            </div>
            <div>
              <h1 className="font-semibold text-text">AI Trading</h1>
              <p className="text-xs text-text-muted">Hedge Fund Edition</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 overflow-y-auto">
          <ul className="flex flex-col gap-2">
            {navItems.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  className={({ isActive }) => `
                    flex items-center gap-3 px-4 py-3 rounded-lg 
                    ${isActive 
                      ? 'bg-primary/20 text-primary border-l-4 border-primary' 
                      : 'text-text-muted hover:text-text hover:bg-surface-hover'}
                    transition-all duration-200
                  `}
                >
                  <item.icon size={20} />
                  <span>{item.label}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        {/* Status */}
        <div className="p-4 border-t border-border">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 bg-success rounded-full animate-pulse"></span>
            <span className="text-xs text-text-muted">Live Trading Active</span>
          </div>
        </div>
      </aside>

      {/* Mobile overlay */}
      {isMobile && sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 z-30"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <main className={`
        flex-1 min-h-0 overflow-y-auto
        ${isMobile 
          ? 'pt-16 px-4'  // Mobile: padding top for hamburger, no left margin
          : 'ml-64'}     // Desktop: left margin for sidebar
        ${isMobile && sidebarOpen ? 'opacity-50 pointer-events-none' : ''}
        transition-opacity duration-200
      `}>
        {/* Demo mode banner */}
        {isUsingFallback && (
          <div className="mt-4 mb-4 bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-4 flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0" />
            <div>
              <p className="text-yellow-500 font-bold text-sm uppercase tracking-wide">Demo Mode Active</p>
              <p className="text-text-muted text-xs">System is operating with simulated data.</p>
            </div>
          </div>
        )}
        <Outlet />
      </main>
    </div>
  );
}