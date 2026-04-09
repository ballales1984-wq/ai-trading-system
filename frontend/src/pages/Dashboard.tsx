import { useQuery } from '@tanstack/react-query';
import { portfolioApi, marketApi, ordersApi } from '../services/api';
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Wallet, Wifi, WifiOff } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { DashboardSkeleton } from '@/components/ui/Skeleton';
import { useMarketData } from '@/hooks/useMarketData';

// ─── Format helpers: definite a livello modulo per evitare ricreazione ad ogni render ─
const formatCurrency = (value: number) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(value);
const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [isMounted, setIsMounted] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Wait for the container to have a non-zero width before mounting charts
    const checkWidth = () => {
      if (containerRef.current && containerRef.current.offsetWidth > 0) {
        setIsMounted(true);
      } else {
        requestAnimationFrame(checkWidth);
      }
    };
    const rafId = requestAnimationFrame(checkWidth);
    return () => cancelAnimationFrame(rafId);
  }, []);

  // ─── Real-time WebSocket data ─────────────────────────────────────────────
  const { prices: livePrices, portfolio: livePortfolio, wsStatus } = useMarketData();

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'positions', label: 'Positions' },
    { id: 'orders', label: 'Orders' },
    { id: 'performance', label: 'Performance' }
  ];

  const { data: dualSummary, isLoading: summaryLoading } = useQuery({
    queryKey: ['portfolio-dual-summary'],
    queryFn: portfolioApi.getDualSummary,
    refetchInterval: 30000,
  });

  const { data: history, error: historyError } = useQuery({
    queryKey: ['portfolio-history'],
    queryFn: () => portfolioApi.getHistory(30),
    refetchInterval: 60000,
  });
  
  // Debug: log API errors
  if (historyError) {
    console.error('History API error:', historyError);
  }

  const { data: markets } = useQuery({
    queryKey: ['market-prices'],
    queryFn: marketApi.getAllPrices,
    refetchInterval: 15000,
  });

  const { data: performance } = useQuery({
    queryKey: ['portfolio-performance'],
    queryFn: portfolioApi.getPerformance,
    refetchInterval: 60000,
  });

  const { data: orders } = useQuery({
    queryKey: ['orders'],
    queryFn: () => ordersApi.list(),
    // 30s: ridotto da 10s — il WS già notifica aggiornamenti in real-time;
    // il polling è solo un fallback quando il WS è chiuso.
    refetchInterval: 30000,
  });

  const { data: positions } = useQuery({
    queryKey: ['portfolio-positions'],
    queryFn: () => portfolioApi.getPositions(),
    refetchInterval: 30000,
  });

  // Prefer WS live data for portfolio summary, fallback to REST
  const summary = livePortfolio ?? (dualSummary?.simulated || { total_value: 0, daily_pnl: 0, unrealized_pnl: 0, num_positions: 0, daily_return_pct: 0 });
  
  // Handle history data - ensure it's always an array
  let historyData: Array<{date: string; value: number}> = [];
  if (history && history.history) {
    historyData = history.history.map((h: any) => ({
      date: new Date(h.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      value: h.value,
    }));
  } else if (history && Array.isArray(history)) {
    // Handle case where API returns array directly
    historyData = history.map((h: any) => ({
      date: new Date(h.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      value: h.value,
    }));
  }
  
  console.log('History raw:', history);
  console.log('History data:', historyData);

  // Merge REST prices with live WS overrides
  const baseMarkets = markets?.markets?.slice(0, 6) || [];
  const marketData = baseMarkets.map((m: any) => {
    const live = livePrices[m.symbol];
    return live ? { ...m, price: live.price, change_pct_24h: live.change_pct_24h } : m;
  });
  const ordersList = (Array.isArray(orders) ? orders : []).slice(0, 8);
  const positionsList = Array.isArray(positions) ? positions : [];

  // Show skeleton while loading (but allow render if we have data)
  const isLoading = summaryLoading || !markets || !performance || !orders || !positions;
  const hasHistory = history && (history.history || Array.isArray(history));
  
  if (isLoading && !hasHistory) {
    return <DashboardSkeleton />;
  }

  return (
    <div className="p-4 md:p-6 space-y-4 md:space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between mt-2 md:mt-0">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Trading Dashboard</h1>
          <p className="text-gray-400">Paper Trading</p>
        </div>
        <div className="flex items-center gap-2">
          {wsStatus === 'open' && (
            <div className="flex items-center gap-2 premium-glass px-3 py-1.5 rounded-full border-green-500/30">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse glow-success"></span>
              <Wifi className="w-4 h-4 text-green-500 drop-shadow-[0_0_5px_rgba(34,197,94,0.5)]" />
              <span className="text-green-500 font-medium text-sm tracking-wide">WS LIVE</span>
            </div>
          )}
          {wsStatus === 'connecting' && (
            <div className="flex items-center gap-2 premium-glass px-3 py-1.5 rounded-full border-yellow-500/30">
              <span className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></span>
              <span className="text-yellow-500 font-medium text-sm tracking-wide">CONNECTING</span>
            </div>
          )}
          {(wsStatus === 'closed' || wsStatus === 'error') && (
            <div className="flex items-center gap-2 premium-glass px-3 py-1.5 rounded-full border-gray-500/30">
              <WifiOff className="w-4 h-4 text-gray-400" />
              <span className="text-gray-400 font-medium text-sm tracking-wide">POLLING</span>
            </div>
          )}
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard title="Total Value" value={formatCurrency(summary?.total_value || 0)} icon={DollarSign} trend={summary?.daily_return_pct} />
        <MetricCard title="Daily P&L" value={formatCurrency(summary?.daily_pnl || 0)} icon={summary?.daily_pnl >= 0 ? TrendingUp : TrendingDown} color={summary?.daily_pnl >= 0 ? 'success' : 'danger'} />
        <MetricCard title="Unrealized P&L" value={formatCurrency(summary?.unrealized_pnl || 0)} icon={summary?.unrealized_pnl >= 0 ? TrendingUp : TrendingDown} color={summary?.unrealized_pnl >= 0 ? 'success' : 'danger'} />
        <MetricCard title="Open Positions" value={String(summary?.num_positions || 0)} icon={Wallet} />
      </div>
      {/* Metric Cards remain fixed at the top */}

      <div className="border-b border-border bg-bg-secondary px-6">
        <div className="flex gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === tab.id ? 'border-primary text-primary' : 'border-transparent text-text-muted hover:text-text'
                }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="p-6">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* MetricCards are already shown above the tabs — no duplication here */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 premium-glass-panel">
                <div className="p-6 border-b border-white/[0.05]">
                  <h3 className="text-lg font-semibold text-text tracking-wide">Portfolio Equity</h3>
                </div>
                <div className="h-72 w-full p-4 relative" style={{ minHeight: '288px' }}>
                  {isMounted && !summaryLoading && historyData.length > 0 ? (
                          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
                      <AreaChart data={historyData}>
                        <defs>
                          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3fb950" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3fb950" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                        <XAxis dataKey="date" stroke="#8b949e" fontSize={11} />
                        <YAxis stroke="#8b949e" fontSize={11} tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
                        <Tooltip contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '8px' }} />
                        <Area type="monotone" dataKey="value" stroke="#3fb950" strokeWidth={2} fillOpacity={1} fill="url(#equityGradient)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-text-muted">No data</div>
                  )}
                </div>
              </div>
              <div className="glass rounded-xl p-6">
                <h3 className="text-lg font-semibold text-text mb-4">Performance</h3>
                <div className="space-y-3">
                  <PerfRow label="Total Return" value={formatPercent(performance?.total_return_pct || 0)} positive={(performance?.total_return_pct || 0) >= 0} />
                  <PerfRow label="Sharpe Ratio" value={String(performance?.sharpe_ratio?.toFixed(2) || '0.00')} />
                  <PerfRow label="Win Rate" value={`${((performance?.win_rate || 0) * 100).toFixed(1)}%`} positive={(performance?.win_rate || 0) > 0.5} />
                  <PerfRow label="Max Drawdown" value={formatPercent(performance?.max_drawdown_pct || 0)} inverted />
                  <PerfRow label="Profit Factor" value={String(performance?.profit_factor?.toFixed(2) || '0.00')} />
                </div>
              </div>
            </div>

            <div className="premium-glass-panel overflow-hidden">
              <div className="px-6 py-5 border-b border-white/[0.05] bg-white/[0.02]">
                <h3 className="text-lg font-semibold text-text tracking-wide flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-primary glow-primary"></span>
                  Live Market Prices
                </h3>
              </div>
              <table className="w-full">
                <thead className="bg-black/20">
                  <tr>
                    <th className="text-left px-6 py-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Symbol</th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Price</th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-text-muted uppercase tracking-wider">24h Change</th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {marketData.map((m: any) => (
                    <tr key={m.symbol} className="border-t border-white/[0.05] hover:bg-white/[0.04] transition-colors">
                      <td className="px-6 py-4 font-semibold text-text flex items-center gap-2">
                        <div className="w-6 h-6 rounded-full bg-white/10 flex items-center justify-center text-xs">
                          {m.symbol.charAt(0)}
                        </div>
                        {m.symbol.replace('USDT', '')}
                      </td>
                      <td className="px-6 py-4 text-right font-mono-num text-text text-lg">
                        <span className={livePrices[m.symbol] ? (livePrices[m.symbol].change_pct_24h > 0 ? 'flash-green' : 'flash-red') : ''}>
                          ${m.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 })}
                        </span>
                      </td>
                      <td className={`px-6 py-4 text-right font-mono-num ${m.change_pct_24h >= 0 ? 'text-success drop-shadow-[0_0_8px_rgba(34,197,94,0.4)]' : 'text-danger drop-shadow-[0_0_8px_rgba(239,68,68,0.4)]'}`}>
                        <div className="bg-black/20 inline-block px-2 py-1 rounded-md">
                          {formatPercent(m.change_pct_24h)}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right font-mono-num text-text-muted">
                        ${(m.volume_24h / 1000000).toFixed(2)}M
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'positions' && (
          <div className="premium-glass-panel overflow-hidden">
            <div className="px-6 py-5 border-b border-white/[0.05] bg-white/[0.02]">
              <h3 className="text-lg font-semibold text-text tracking-wide">Open Positions ({positionsList.length})</h3>
            </div>
            {positionsList.length > 0 ? (
              <table className="w-full">
                <thead className="bg-black/20">
                  <tr>
                    <th className="text-left px-6 py-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Symbol</th>
                    <th className="text-left px-6 py-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Side</th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Qty</th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Entry</th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-text-muted uppercase tracking-wider">Current</th>
                    <th className="text-right px-6 py-4 text-xs font-semibold text-text-muted uppercase tracking-wider">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {positionsList.map((p: any) => {
                    const pnl = p.unrealized_pnl || 0;
                    const pnlPercent = p.entry_price > 0 ? ((p.current_price - p.entry_price) / p.entry_price) * 100 : 0;
                    return (
                      <tr key={p.position_id} className="border-t border-white/[0.05] hover:bg-white/[0.04]">
                        <td className="px-6 py-4 font-semibold text-text">{p.symbol}</td>
                        <td className="px-6 py-4">
                          <span className={`px-2 py-1 rounded text-xs font-bold tracking-wider ${p.side === 'LONG' ? 'bg-success/10 text-success border border-success/30' : 'bg-danger/10 text-danger border border-danger/30'}`}>
                            {p.side}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right text-text font-mono-num">{p.quantity.toFixed(4)}</td>
                        <td className="px-6 py-4 text-right text-text-muted font-mono-num">{formatCurrency(p.entry_price)}</td>
                        <td className="px-6 py-4 text-right text-text font-mono-num">{formatCurrency(p.current_price)}</td>
                        <td className={`px-6 py-4 text-right font-medium font-mono-num ${pnl >= 0 ? 'text-success drop-shadow-[0_0_5px_rgba(34,197,94,0.3)]' : 'text-danger drop-shadow-[0_0_5px_rgba(239,68,68,0.3)]'}`}>
                          <div className="text-lg">{formatCurrency(pnl)}</div>
                          <div className="text-xs opacity-80">{pnlPercent >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%</div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : (
              <div className="p-12 text-center text-text-muted">No open positions</div>
            )}
          </div>
        )}

        {activeTab === 'orders' && (
          <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-border">
              <h3 className="text-lg font-semibold text-text">Recent Orders ({ordersList.length})</h3>
            </div>
            {ordersList.length > 0 ? (
              <table className="w-full">
                <thead className="bg-bg-tertiary">
                  <tr>
                    <th className="text-left px-6 py-3 text-sm font-medium text-text-muted">Time</th>
                    <th className="text-left px-6 py-3 text-sm font-medium text-text-muted">Symbol</th>
                    <th className="text-left px-6 py-3 text-sm font-medium text-text-muted">Side</th>
                    <th className="text-right px-6 py-3 text-sm font-medium text-text-muted">Qty</th>
                    <th className="text-center px-6 py-3 text-sm font-medium text-text-muted">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {ordersList.map((o: any) => (
                    <tr key={o.order_id} className="border-t border-border">
                      <td className="px-6 py-4 text-text-muted text-sm">
                        {new Date(o.created_at).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                      </td>
                      <td className="px-6 py-4 font-medium text-text">{o.symbol}</td>
                      <td className="px-6 py-4">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${o.side === 'BUY' ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'}`}>
                          {o.side}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right text-text font-mono">{o.quantity.toFixed(4)}</td>
                      <td className="px-6 py-4 text-center">
                        <StatusBadge status={o.status} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="p-12 text-center text-text-muted">No orders yet</div>
            )}
          </div>
        )}

        {activeTab === 'performance' && (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <PerfCard label="Total Return" value={formatPercent(performance?.total_return_pct || 0)} positive={(performance?.total_return_pct || 0) >= 0} />
            <PerfCard label="Sharpe Ratio" value={String(performance?.sharpe_ratio?.toFixed(2) || '0.00')} />
            <PerfCard label="Sortino Ratio" value={String(performance?.sortino_ratio?.toFixed(2) || '0.00')} />
            <PerfCard label="Win Rate" value={`${((performance?.win_rate || 0) * 100).toFixed(1)}%`} positive={(performance?.win_rate || 0) > 0.5} />
            <PerfCard label="Max Drawdown" value={formatPercent(performance?.max_drawdown_pct || 0)} inverted />
            <PerfCard label="Profit Factor" value={String(performance?.profit_factor?.toFixed(2) || '0.00')} positive={(performance?.profit_factor || 0) > 1.5} />
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({ title, value, icon: Icon, trend, color }: { title: string; value: string; icon: any; trend?: number; color?: string }) {
  const isPositive = color === 'success' || (trend !== undefined && trend >= 0);
  const isNegative = color === 'danger' || (trend !== undefined && trend < 0);

  return (
    <div className="premium-glass-panel p-5 premium-glass-hover relative overflow-hidden group">
      {/* Decorative gradient blob */}
      <div className={`absolute -right-6 -top-6 w-24 h-24 rounded-full blur-2xl opacity-20 transition-opacity group-hover:opacity-40 
        ${isPositive ? 'bg-success' : isNegative ? 'bg-danger' : 'bg-primary'}`}></div>

      <div className="flex items-center justify-between mb-3 relative z-10">
        <span className="text-sm font-medium text-text-muted uppercase tracking-wider">{title}</span>
        <div className={`p-2 rounded-lg bg-black/30 border ${isPositive ? 'border-success/30 text-success glow-success' : isNegative ? 'border-danger/30 text-danger glow-danger' : 'border-primary/30 text-primary glow-primary'}`}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <div className={`text-2xl lg:text-3xl font-bold font-mono-num relative z-10 ${isPositive ? 'text-success' : isNegative ? 'text-danger' : 'text-text'}`}>
        {value}
      </div>
      {trend !== undefined && (
        <div className={`text-sm mt-2 font-medium flex items-center gap-1 relative z-10 ${trend >= 0 ? 'text-success' : 'text-danger'}`}>
          {trend >= 0 ? '▲' : '▼'} {Math.abs(trend).toFixed(2)}% <span className="text-text-muted ml-1 font-normal text-xs uppercase">vs ieri</span>
        </div>
      )}
    </div>
  );
}

function PerfRow({ label, value, positive, inverted }: { label: string; value: string; positive?: boolean; inverted?: boolean }) {
  let valueColor = 'text-text';
  if (positive !== undefined) valueColor = positive ? 'text-success' : 'text-danger';
  if (inverted) valueColor = value.startsWith('+') ? 'text-danger' : 'text-success';

  return (
    <div className="glass py-2 border-b border-border/50">
      <div className="flex justify-between items-center">
        <span className="text-sm text-text-muted">{label}</span>
        <span className={`font-semibold ${valueColor}`}>{value}</span>
      </div>
    </div>
  );
}

function PerfCard({ label, value, positive, inverted }: { label: string; value: string; positive?: boolean; inverted?: boolean }) {
  let valueColor = 'text-text';
  if (positive !== undefined) valueColor = positive ? 'text-success' : 'text-danger';
  if (inverted) valueColor = value.startsWith('+') ? 'text-danger' : 'text-success';

  return (
    <div className="glass rounded-xl p-4">
      <div className="text-sm text-text-muted mb-1">{label}</div>
      <div className={`text-2xl font-bold ${valueColor}`}>{value}</div>
    </div>
  );
}

