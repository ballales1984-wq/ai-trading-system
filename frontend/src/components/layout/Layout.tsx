import { useState } from 'react';
import { Outlet, NavLink, Link } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { LayoutDashboard, PieChart, TrendingUp, ClipboardList, Bot, Menu, X, Megaphone } from 'lucide-react';
import { ToastContainer, useToast } from '../ui/Toast';
import { emergencyApi } from '../../services/api';

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/portfolio', icon: PieChart, label: 'Portfolio' },
  { to: '/market', icon: TrendingUp, label: 'Market' },
  { to: '/orders', icon: ClipboardList, label: 'Orders' },
];

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showEmergencyModal, setShowEmergencyModal] = useState(false);
  const [adminKeyInput, setAdminKeyInput] = useState('');
  const [emergencyReason, setEmergencyReason] = useState('');
  const { toasts, removeToast, toast } = useToast();
  const queryClient = useQueryClient();

  const { data: emergencyStatus } = useQuery({
    queryKey: ['emergency-status'],
    queryFn: emergencyApi.getStatus,
    refetchInterval: 5000,
  });

  const activateEmergency = useMutation({
    mutationFn: ({ reason, adminKey }: { reason: string; adminKey: string }) =>
      emergencyApi.activate(reason, adminKey),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emergency-status'] });
      toast.warning('Emergency stop activated');
    },
    onError: () => {
      toast.error('Failed to activate emergency stop');
    },
  });

  const deactivateEmergency = useMutation({
    mutationFn: ({ reason, adminKey }: { reason: string; adminKey: string }) =>
      emergencyApi.deactivate(reason, adminKey),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emergency-status'] });
      toast.success('Trading resumed');
    },
    onError: () => {
      toast.error('Failed to resume trading');
    },
  });

  const isTradingHalted = Boolean(emergencyStatus?.trading_halted);
  const isMutating = activateEmergency.isPending || deactivateEmergency.isPending;

  const getStoredAdminKey = (): string => {
    const fromStorage = window.localStorage.getItem('admin_emergency_key');
    return fromStorage || '';
  };

  const openEmergencyModal = () => {
    setAdminKeyInput(getStoredAdminKey());
    setEmergencyReason(
      isTradingHalted ? 'Manual resume from dashboard' : 'Manual emergency stop from dashboard'
    );
    setShowEmergencyModal(true);
  };

  const closeEmergencyModal = () => {
    if (isMutating) return;
    setShowEmergencyModal(false);
  };

  const submitEmergencyAction = () => {
    const adminKey = adminKeyInput.trim();
    if (!adminKey) return;

    window.localStorage.setItem('admin_emergency_key', adminKey);
    const reason = emergencyReason.trim() || (isTradingHalted ? 'Manual resume from dashboard' : 'Manual emergency stop from dashboard');

    if (isTradingHalted) {
      deactivateEmergency.mutate(
        { reason, adminKey },
        { onSuccess: () => setShowEmergencyModal(false) }
      );
    } else {
      activateEmergency.mutate(
        { reason, adminKey },
        { onSuccess: () => setShowEmergencyModal(false) }
      );
    }
  };

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
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
                  <Bot className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-text">AI Trading</h1>
                  <p className="text-xs text-text-muted">Operator Console</p>
                </div>
              </div>
              <Link to="/" className="inline-flex items-center gap-1 text-xs text-text-muted hover:text-text">
                <Megaphone className="w-3 h-3" />
                Marketing
              </Link>
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
            <div className="flex items-center justify-between gap-2 text-sm">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${isTradingHalted ? 'bg-danger' : 'bg-success'} live-indicator`} />
                <span className="text-text-muted">
                  {isTradingHalted ? 'Emergency Stop Active' : 'Live Mode Active'}
                </span>
              </div>
              <button
                onClick={openEmergencyModal}
                disabled={isMutating}
                className={`px-2 py-1 rounded border text-xs disabled:opacity-50 ${
                  isTradingHalted
                    ? 'border-success/50 text-success hover:bg-success/10'
                    : 'border-danger/50 text-danger hover:bg-danger/10'
                }`}
                title={isTradingHalted ? 'Resume trading' : 'Emergency stop trading'}
              >
                {isMutating ? '...' : isTradingHalted ? 'Resume' : 'STOP'}
              </button>
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
          <div className="mx-6 mt-4 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning">
            Trading involves risk. This platform provides tools and analytics, not investment advice.
          </div>
          <Outlet />
          <footer className="px-6 py-6 text-xs text-text-muted border-t border-border mt-8">
            <div className="flex flex-wrap items-center gap-4">
              <Link className="hover:text-text" to="/legal/terms">Terms</Link>
              <Link className="hover:text-text" to="/legal/privacy">Privacy</Link>
              <Link className="hover:text-text" to="/legal/risk">Risk Disclosure</Link>
              <span>Educational/demo mode may use simulated data.</span>
            </div>
          </footer>
        </main>
      </div>

      {/* Toast Notifications */}
      <ToastContainer toasts={toasts} removeToast={removeToast} />

      {showEmergencyModal && (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4">
          <div className="w-full max-w-md rounded-lg border border-border bg-surface p-5">
            <h2 className="text-lg font-semibold text-text">
              {isTradingHalted ? 'Resume Trading' : 'Emergency Stop'}
            </h2>
            <p className="mt-2 text-sm text-text-muted">
              {isTradingHalted
                ? 'Re-enable BUY and SELL actions for all users.'
                : 'Immediately block BUY and SELL actions for all users.'}
            </p>
            <div className="mt-4 space-y-3">
              <div>
                <label htmlFor="admin-key" className="mb-1 block text-sm text-text-muted">Admin key</label>
                <input
                  id="admin-key"
                  type="password"
                  value={adminKeyInput}
                  onChange={(e) => setAdminKeyInput(e.target.value)}
                  className="w-full rounded border border-border bg-background px-3 py-2 text-text outline-none focus:border-primary"
                />
              </div>
              <div>
                <label htmlFor="emergency-reason" className="mb-1 block text-sm text-text-muted">Reason</label>
                <input
                  id="emergency-reason"
                  type="text"
                  value={emergencyReason}
                  onChange={(e) => setEmergencyReason(e.target.value)}
                  className="w-full rounded border border-border bg-background px-3 py-2 text-text outline-none focus:border-primary"
                />
              </div>
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={closeEmergencyModal}
                disabled={isMutating}
                className="rounded border border-border px-3 py-2 text-sm text-text-muted hover:text-text disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                onClick={submitEmergencyAction}
                disabled={isMutating || !adminKeyInput.trim()}
                className={`rounded px-3 py-2 text-sm font-semibold text-white disabled:opacity-60 ${
                  isTradingHalted ? 'bg-success hover:bg-success/80' : 'bg-danger hover:bg-danger/80'
                }`}
              >
                {isMutating ? 'Submitting...' : isTradingHalted ? 'Resume trading' : 'Activate stop'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
