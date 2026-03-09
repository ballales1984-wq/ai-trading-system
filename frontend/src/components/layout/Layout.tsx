import { useState } from 'react';
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

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#0d1117' }}>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        style={{
          position: 'fixed',
          top: 16,
          left: 16,
          zIndex: 50,
          padding: 8,
          backgroundColor: '#161b22',
          border: '1px solid #30363d',
          borderRadius: 8,
          display: 'none',
        }}
      >
        {sidebarOpen ? <X size={24} color="#c9d1d9" /> : <Menu size={24} color="#c9d1d9" />}
      </button>

      {/* Sidebar */}
      <aside
        style={{
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
          width: 260,
          backgroundColor: 'rgba(22, 27, 34, 0.95)',
          backdropFilter: 'blur(20px)',
          borderRight: '1px solid #30363d',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 40,
          transition: 'transform 0.3s ease',
        }}
      >
        {/* Logo */}
        <div style={{ padding: 20, borderBottom: '1px solid #30363d' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div
              style={{
                width: 40,
                height: 40,
                background: 'linear-gradient(135deg, #58a6ff 0%, #a371f7 100%)',
                borderRadius: 10,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Bot size={24} color="#fff" />
            </div>
            <div>
              <h1 style={{ fontSize: 16, fontWeight: 700, color: '#c9d1d9', margin: 0 }}>AI Trading</h1>
              <p style={{ fontSize: 11, color: '#8b949e', margin: '2px 0 0' }}>Hedge Fund Edition</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav style={{ flex: 1, padding: 16 }}>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {navItems.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  onClick={() => setSidebarOpen(false)}
                  style={({ isActive }) => ({
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: '12px 16px',
                    borderRadius: 8,
                    color: isActive ? '#58a6ff' : '#8b949e',
                    backgroundColor: isActive ? 'rgba(88, 166, 255, 0.1)' : 'transparent',
                    textDecoration: 'none',
                    fontSize: 14,
                    fontWeight: 500,
                    transition: 'all 0.2s ease',
                  })}
                >
                  <item.icon size={20} />
                  <span>{item.label}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        {/* Status */}
        <div style={{ padding: 16, borderTop: '1px solid #30363d' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span
              style={{
                width: 8,
                height: 8,
                backgroundColor: '#3fb950',
                borderRadius: '50%',
                animation: 'pulse 2s infinite',
              }}
            />
            <span style={{ fontSize: 12, color: '#8b949e' }}>Live Trading Active</span>
          </div>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            zIndex: 30,
          }}
        />
      )}

      {/* Main Content */}
      <main style={{ flex: 1, marginLeft: 260, overflow: 'auto' }}>
        <Outlet />
      </main>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}

