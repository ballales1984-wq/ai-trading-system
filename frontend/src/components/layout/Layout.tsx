import { useState } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { LayoutDashboard, PieChart, TrendingUp, ClipboardList, Bot, Menu, X } from 'lucide-react';
import { ToastContainer, useToast } from '../ui/Toast';

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/portfolio', icon: PieChart, label: 'Portfolio' },
  { to: '/market', icon: TrendingUp, label: 'Market' },
  { to: '/orders', icon: ClipboardList, label: 'Orders' },
];

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { toasts, removeToast } = useToast();

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <div className="flex flex-1">
        {/* Mobile Menu Button */}
        <button
          className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-surface rounded-lg border border-border"
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          {sidebarOpen ? (
            <X className="w-6 h-6 text-text" />
          ) : (
            <Menu className="w-6 h-6 text-text" />
          )}
        </button>

        {/* Sidebar */}
        <aside
          className={`
            fixed lg:static inset-y-0 left-0 z-40 w-64 bg-surface border-r border-border flex flex-col
            transform transition-transform duration-300 ease-in-out
            ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          `}
        >
          {/* Logo */}
          <div className="p-4 border-b border-border mt-12 lg:mt-0">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
                <Bot className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-text">AI Trading</h1>
                <p className="text-xs text-text-muted">Hedge Fund Edition</p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4">
            <ul className="space-y-2">
              {navItems.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    onClick={() => setSidebarOpen(false)}
                    className={({ isActive }) =>
                      `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                        isActive
                          ? 'bg-primary/10 text-primary'
                          : 'text-text-muted hover:bg-surface hover:text-text'
                      }`
                    }
                  >
                    <item.icon className="w-5 h-5" />
                    <span className="font-medium">{item.label}</span>
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>

          {/* Status */}
          <div className="p-4 border-t border-border">
            <div className="flex items-center gap-2 text-sm">
              <span className="w-2 h-2 bg-success rounded-full live-indicator" />
              <span className="text-text-muted">Live Mode Active</span>
            </div>
          </div>
        </aside>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-30 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>

      {/* Toast Notifications */}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </div>
  );
}
