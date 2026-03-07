import { useState } from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, 
  PieChart, 
  TrendingUp, 
  ClipboardList, 
  Bot, 
  Menu, 
  X,
  Settings,
  LogOut,
  ChevronRight
} from 'lucide-react';
import { ToastContainer, useToast } from '../ui/Toast';

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard', description: 'Overview & Analytics' },
  { to: '/portfolio', icon: PieChart, label: 'Portfolio', description: 'Holdings & Allocation' },
  { to: '/market', icon: TrendingUp, label: 'Market', description: 'Live Market Data' },
  { to: '/orders', icon: ClipboardList, label: 'Orders', description: 'Trade History' },
];

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const { toasts, removeToast } = useToast();
  const location = useLocation();

  const currentPage = navItems.find(item => item.to === location.pathname);

  return (
    <div className="min-h-screen bg-bg-primary flex">
      {/* Sidebar */}
      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-40 
          bg-bg-secondary border-r border-border flex flex-col
          transition-all duration-300 ease-in-out
          ${collapsed ? 'w-20' : 'w-64'}
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Logo */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary/20 rounded-xl flex items-center justify-center flex-shrink-0">
              <Bot className="w-6 h-6 text-primary" />
            </div>
            {!collapsed && (
              <div className="overflow-hidden">
                <h1 className="text-lg font-bold text-text truncate">AI Trading</h1>
                <p className="text-xs text-text-muted truncate">Hedge Fund Edition</p>
              </div>
            )}
          </div>
        </div>

        {/* Collapse Button (desktop) */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="hidden lg:flex items-center justify-center p-2 mx-auto text-text-muted hover:text-text hover:bg-bg-tertiary rounded-lg transition-colors"
        >
          <ChevronRight className={`w-5 h-5 transition-transform ${collapsed ? 'rotate-180' : ''}`} />
        </button>

        {/* Navigation */}
        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={() => setSidebarOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-3 rounded-xl transition-all duration-200 ${
                  isActive
                    ? 'bg-primary/15 text-primary border border-primary/30'
                    : 'text-text-muted hover:bg-bg-tertiary hover:text-text border border-transparent'
                } ${collapsed ? 'justify-center' : ''}`
              }
            >
              <item.icon className="w-5 h-5 flex-shrink-0" />
              {!collapsed && (
                <div className="overflow-hidden">
                  <div className="font-medium text-sm truncate">{item.label}</div>
                  <div className="text-xs text-text-muted truncate">{item.description}</div>
                </div>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Bottom Actions */}
        <div className="p-3 border-t border-border space-y-1">
          <button
            className={`flex items-center gap-3 px-3 py-3 rounded-xl text-text-muted hover:bg-bg-tertiary hover:text-text transition-colors w-full ${
              collapsed ? 'justify-center' : ''
            }`}
          >
            <Settings className="w-5 h-5 flex-shrink-0" />
            {!collapsed && <span className="text-sm font-medium">Settings</span>}
          </button>
          <button
            className={`flex items-center gap-3 px-3 py-3 rounded-xl text-danger hover:bg-danger/10 transition-colors w-full ${
              collapsed ? 'justify-center' : ''
            }`}
          >
            <LogOut className="w-5 h-5 flex-shrink-0" />
            {!collapsed && <span className="text-sm font-medium">Logout</span>}
          </button>
        </div>

        {/* Status */}
        <div className="p-4 border-t border-border">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 bg-success rounded-full animate-pulse" />
            {!collapsed && (
              <span className="text-xs text-text-muted">Live Trading Active</span>
            )}
          </div>
        </div>
      </aside>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-screen">
        {/* Top Header */}
        <header className="sticky top-0 z-20 bg-bg-secondary/95 backdrop-blur-sm border-b border-border px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* Mobile Menu Button */}
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden p-2 bg-bg-tertiary rounded-lg text-text hover:bg-border transition-colors"
              >
                {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
              
              {/* Page Title */}
              <div>
                <h1 className="text-lg font-semibold text-text">
                  {currentPage?.label || 'Dashboard'}
                </h1>
                {currentPage && (
                  <p className="text-xs text-text-muted">{currentPage.description}</p>
                )}
              </div>
            </div>

            {/* Right side - Time & Status */}
            <div className="flex items-center gap-4">
              <div className="text-right hidden sm:block">
                <div className="text-xs text-text-muted">
                  {new Date().toLocaleDateString('en-US', { 
                    weekday: 'short', 
                    month: 'short', 
                    day: 'numeric' 
                  })}
                </div>
                <div className="text-xs text-text-muted">
                  {new Date().toLocaleTimeString('en-US', { 
                    hour: '2-digit', 
                    minute: '2-digit',
                    timeZoneName: 'short'
                  })}
                </div>
              </div>
              <div className="px-3 py-1.5 bg-success/10 border border-success/20 rounded-full">
                <span className="text-xs font-medium text-success">Paper Trading</span>
              </div>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-4 lg:p-6">
          <Outlet />
        </main>
      </div>

      {/* Toast Notifications */}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </div>
  );
}
