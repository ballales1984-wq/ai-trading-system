import { useState, useEffect, useCallback } from 'react';
  import { Outlet, NavLink } from 'react-router-dom';
  import { LayoutDashboard, PieChart, TrendingUp, ClipboardList, Menu, X, Bot, FileText } from 'lucide-react';

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/portfolio', icon: PieChart, label: 'Portfolio' },
  { to: '/market', icon: TrendingUp, label: 'Market' },
  { to: '/orders', icon: ClipboardList, label: 'Orders' },
  { to: '/news', icon: FileText, label: 'News' },
];

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  // Detect mobile device
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Close sidebar on route change (mobile)
  useEffect(() => {
    if (isMobile) {
      setSidebarOpen(false);
    }
  }, [isMobile]);

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
        <Outlet />
      </main>
    </div>
  );
}

