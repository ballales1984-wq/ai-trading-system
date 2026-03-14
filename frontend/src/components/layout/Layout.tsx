import { useState, useEffect } from 'react';
  import { Outlet, NavLink } from 'react-router-dom';
import { LayoutDashboard, PieChart, TrendingUp, ClipboardList, Menu, X, Bot, FileText, Target, Shield, Settings, AlertTriangle } from 'lucide-react';
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
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [touchStartX, setTouchStartX] = useState(0);
  const [touchEndX, setTouchEndX] = useState(0);

  // Global check for backend connectivity to show "Demo Mode" banner
  const { data: prices } = useQuery({
    queryKey: ['market-prices'],
    queryFn: marketApi.getAllPrices,
    retry: 1,
    staleTime: 60000,
  });

  const isUsingFallback = !prices?.markets || prices.markets.length === 0;

  // Detect mobile device
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    // Add touch event listeners for swipe gestures on mobile
    if (isMobile) {
      const handleTouchStart = (e: TouchEvent) => {
        setTouchStartX(e.touches[0].clientX);
      };
      
      const handleTouchEnd = (e: TouchEvent) => {
        setTouchEndX(e.changedTouches[0].clientX);
        handleSwipeGesture();
      };
      
      const handleSwipeGesture = () => {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;
        
        // Swipe right to open sidebar
        if (diff < -swipeThreshold) {
          setSidebarOpen(true);
        }
        // Swipe left to close sidebar
        if (diff > swipeThreshold) {
          setSidebarOpen(false);
        }
      };
      
      window.addEventListener('touchstart', handleTouchStart);
      window.addEventListener('touchend', handleTouchEnd);
      
      return () => {
        window.removeEventListener('resize', checkMobile);
        window.removeEventListener('touchstart', handleTouchStart);
        window.removeEventListener('touchend', handleTouchEnd);
      };
    }
    
    return () => window.removeEventListener('resize', checkMobile);
  }, [isMobile, touchStartX, touchEndX]);

  // Close sidebar on route change (mobile)
  useEffect(() => {
    if (isMobile) {
      setSidebarOpen(false);
    }
  }, [isMobile]);

  // Handle escape key to close sidebar on mobile
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isMobile && sidebarOpen) {
        setSidebarOpen(false);
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isMobile, sidebarOpen]);

  return (
    <div className="flex min-h-screen bg-background">
      {/* Mobile Menu Button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className={`fixed top-4 left-4 z-50 flex h-10 w-10 items-center justify-center rounded-md 
                   bg-surface border border-border hover:bg-surface-hover 
                   ${isMobile ? 'block' : 'hidden'}`}
        aria-label={sidebarOpen ? 'Close menu' : 'Open menu'}
      >
        {sidebarOpen ? <X size={20} color="#c9d1d9" /> : <Menu size={20} color="#c9d1d9" />}
      </button>

      {/* Sidebar */}
      <aside className={`fixed left-0 top-0 bottom-0 w-64 
                         ${isMobile ? 'translate-x-0' : 'translate-x-0'}
                         ${isMobile && sidebarOpen ? 'translate-x-0' : isMobile ? '-translate-x-full' : 'translate-x-0'}
                         transition-transform duration-300 ease-in-out z-40 flex flex-col`}
        >
        {/* Logo */}
        <div className="glass p-4 mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-tr from-primary to-purple">
              <Bot size={20} color="#fff" />
            </div>
            <div>
              <h1 className="font-semibold text-text">AI Trading</h1>
              <p className="text-xs text-text-muted">Hedge Fund Edition</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="flex flex-col gap-2">
            {navItems.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  onClick={() => setSidebarOpen(false)}
                  className={({ isActive }) => `
                    flex items-center gap-3 px-4 py-3 rounded-lg 
                    ${isActive ? 'bg-primary/20 text-primary border-l-4 border-primary' : 'text-text-muted hover:text-text'}
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
        <div className="glass p-4 mt-auto">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 bg-success rounded-full animate-pulse"></span>
            <span className="text-xs text-text-muted">Live Trading Active</span>
          </div>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {sidebarOpen && isMobile && (
        <div
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 bg-black/50 z-30"
        />
      )}

      {/* Main Content */}
      <main className={`flex-1 min-h-0 
                       ${isMobile ? 'ml-0 mt-16' : 'ml-64'} 
                       overflow-y-auto`}
      >
        {/* Global Fallback Data Warning */}
        {isUsingFallback && (
          <div className="m-6 mb-2 bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-4 flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0" />
            <div>
              <p className="text-yellow-500 font-bold text-sm uppercase tracking-wide">Demo Mode Active</p>
              <p className="text-text-muted text-xs">System is operating with simulated data because the live terminal is unreachable.</p>
            </div>
          </div>
        )}
        <Outlet />
      </main>
    </div>
  );
}

