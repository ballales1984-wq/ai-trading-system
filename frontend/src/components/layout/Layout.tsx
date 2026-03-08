
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
  ChevronLeft,
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
    <div className="min-h-screen flex bg-[var(--bg-primary)]">
      {/* Sidebar */}
      <aside
        className={`
          fixed inset-y-0 left-0 z-50 
          bg-[var(--bg-secondary)] border-r border-[var(--border-color)] flex flex-col
          transition-all duration-300 ease-in-out
          ${collapsed ? 'w-16' : 'w-64'}
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Logo */}
        <div className="p-4 border-b border-[var(--border-color)]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-[var(--accent-blue)]/20 rounded-xl flex items-center justify-center flex-shrink-0">
              <Bot className="w-6 h-6 text-[var(--accent-blue)]" />
            </div>
            {!collapsed && (
              <div className="flex flex-col">
                <h1 className="text-lg font-bold text-[var(--text-primary)]">AI Trading</h1>
                <p className="text-xs text-[var(--text-secondary)]">Hedge Fund Edition</p>
              </div>
            )}
          </div>
        </div>

        {/* Collapse Button */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="hidden lg:flex items-center justify-center p-2 mx-auto my-1 text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-lg transition-colors"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
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
                    ? 'bg-[var(--accent-blue)]/15 text-[var(--accent-blue)] border border-[var(--accent-blue)]/30'
                    : 'text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)] border border-transparent'
                } ${collapsed ? 'justify-center' : ''}`
              }
            >
              <item.icon className="w-5 h-5 flex-shrink-0" />
              {!collapsed && (
                <div className="flex flex-col">
                  <span className="font-medium text-sm">{item.label}</span>
                  <span className="text-xs text-[var(--text-secondary)]">{item.description}</span>
                </div>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Bottom Actions */}
        <div className="p-3 border-t border-[var(--border-color)] space-y-1">
          <button
            className={`flex items-center gap-3 px-3 py-3 rounded-xl text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)] transition-colors w-full ${
              collapsed ? 'justify-center' : ''
            }`}
          >
            <Settings className="w-5 h-5 flex-shrink-0" />
            {!collapsed && <span className="text-sm font-medium">Settings</span>}
          </button>
          <button
            className={`flex items-center gap-3 px-3 py-3 rounded-xl text-[var(--accent-red)] hover:bg-[var(--accent-red)]/10 transition-colors w-full ${
              collapsed ? 'justify-center' : ''
            }`}
          >
            <LogOut className="w-5 h-5 flex-shrink-0" />
            {!collapsed && <span className="text-sm font-medium">Logout</span>}
          </button>
        </div>

        {/* Status */}
        <div className="p-4 border-t border-[var(--border-color)]">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 bg-[var(--accent-green)] rounded-full animate-pulse" />
            {!collapsed && (
              <span className="text-xs text-[var(--text-secondary)]">Live Trading Active</span>
            )}
          </div>
        </div>
      </aside>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-screen lg:ml-64">
        {/* Top Header */}
        <header className="sticky top-0 z-30 bg-[var(--bg-secondary)]/95 backdrop-blur-sm border-b border-[var(--border-color)] px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* Mobile Menu Button */}
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden p-2 bg-[var(--bg-tertiary)] rounded-lg text-[var(--text-primary)] hover:bg-[var(--border-color)] transition-colors"
              >
                {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
              
              {/* Page Title */}
              <div>
                <h1 className="text-lg font-semibold text-[var(--text-primary)]">
                  {currentPage?.label || 'Dashboard'}
                </h1>
                {currentPage && (
                  <p className="text-xs text-[var(--text-secondary)]">{currentPage.description}</p>
                )}
              </div>
            </div>

            {/* Right side - Time & Status */}
            <div className="flex items-center gap-4">
              <div className="text-right hidden sm:block">
                <div className="text-xs text-[var(--text-secondary)]">
                  {new Date().toLocaleDateString('en-US', { 
                    weekday: 'short', 
                    month: 'short', 
                    day: 'numeric' 
                  })}
                </div>
                <div className="text-xs text-[var(--text-secondary)]">
                  {new Date().toLocaleTimeString('en-US', { 
                    hour: '2-digit', 
                    minute: '2-digit',
                    timeZoneName: 'short'
                  })}
                </div>
              </div>
              <div className="px-3 py-1.5 bg-[var(--accent-green)]/10 border border-[var(--accent-green)]/20 rounded-full">
                <span className="text-xs font-medium text-[var(--accent-green)]">Paper Trading</span>
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

