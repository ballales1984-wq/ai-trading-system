import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { portfolioApi, marketApi, ordersApi } from '../services/api';
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Wallet } from 'lucide-react';

type TabType = 'overview' | 'positions' | 'orders' | 'performance';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  
  const { data: dualSummary, isLoading: summaryLoading } = useQuery({
    queryKey: ['portfolio-dual-summary'],
    queryFn: portfolioApi.getDualSummary,
    refetchInterval: 30000,
  });

  const { data: history } = useQuery({
    queryKey: ['portfolio-history'],
    queryFn: () => portfolioApi.getHistory(30),
    refetchInterval: 60000,
  });

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
    refetchInterval: 10000,
  });

  const { data: positions } = useQuery({
    queryKey: ['portfolio-positions'],
    queryFn: () => portfolioApi.getPositions(),
    refetchInterval: 30000,
  });

  const summary = dualSummary?.simulated || { total_value: 0, daily_pnl: 0, unrealized_pnl: 0, num_positions: 0, daily_return_pct: 0 };
  const historyData = history?.history?.map((h: any) => ({
    date: new Date(h.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    value: h.value,
  })) || [];

  const marketData = markets?.markets?.slice(0, 6) || [];
  const ordersList = (Array.isArray(orders) ? orders : []).slice(0, 8);
  const positionsList = Array.isArray(positions) ? positions : [];

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(value);
  };

  const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;

  const tabs = [
    { id: 'overview' as const, label: 'Overview' },
    { id: 'positions' as const, label: 'Positions' },
    { id: 'orders' as const, label: 'Orders' },
    { id: 'performance' as const, label: 'Performance' },
  ];

  if (summaryLoading && !dualSummary) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg-primary">
      <div className="border-b border-border bg-bg-secondary px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-text">Trading Dashboard</h1>
            <p className="text-sm text-text-muted">Paper Trading</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-success rounded-full animate-pulse"></span>
            <span className="text-sm text-success font-medium">Live</span>
          </div>
        </div>
      </div>

      <div className="border-b border-border bg-bg-secondary px-6">
        <div className="flex gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id ? 'border-primary text-primary' : 'border-transparent text-text-muted hover:text-text'
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
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard title="Total Value" value={formatCurrency(summary?.total_value || 0)} icon={DollarSign} trend={summary?.daily_return_pct} />
              <MetricCard title="Daily P&L" value={formatCurrency(summary?.daily_pnl || 0)} icon={summary?.daily_pnl >= 0 ? TrendingUp : TrendingDown} color={summary?.daily_pnl >= 0 ? 'success' : 'danger'} />
              <MetricCard title="Unrealized P&L" value={formatCurrency(summary?.unrealized_pnl || 0)} icon={summary?.unrealized_pnl >= 0 ? TrendingUp : TrendingDown} color={summary?.unrealized_pnl >= 0 ? 'success' : 'danger'} />
              <MetricCard title="Open Positions" value={String(summary?.num_positions || 0)} icon={Wallet} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 bg-bg-secondary border border-border rounded-xl p-6">
                <h3 className="text-lg font-semibold text-text mb-4">Portfolio Equity</h3>
<div className="h-72 w-full min-h-[280px]">
                  {!summaryLoading && historyData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={historyData}>
                        <defs>
                          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3fb950" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#3fb950" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                        <XAxis dataKey="date" stroke="#8b949e" fontSize={11} />
                        <YAxis stroke="#8b949e" fontSize={11} tickFormatter={(v: number) => `$${(v/1000).toFixed(0)}k`} />
                        <Tooltip contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '8px' }} />
                        <Area type="monotone" dataKey="value" stroke="#3fb950" strokeWidth={2} fillOpacity={1} fill="url(#equityGradient)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-text-muted">No data</div>
                  )}
                </div>
              </div>

              <div className="bg-bg-secondary border border-border rounded-xl p-6">
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

            <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
              <div className="px-6 py-4 border-b border-border">
                <h3 className="text-lg font-semibold text-text">Live Market Prices</h3>
              </div>
              <table className="w-full">
                <thead className="bg-bg-tertiary">
                  <tr>
                    <th className="text-left px-6 py-3 text-sm font-medium text-text-muted">Symbol</th>
                    <th className="text-right px-6 py-3 text-sm font-medium text-text-muted">Price</th>
                    <th className="text-right px-6 py-3 text-sm font-medium text-text-muted">24h Change</th>
                    <th className="text-right px-6 py-3 text-sm font-medium text-text-muted">Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {marketData.map((m: any) => (
                    <tr key={m.symbol} className="border-t border-border hover:bg-bg-tertiary/50">
                      <td className="px-6 py-3 font-medium text-text">{m.symbol.replace('USDT', '')}</td>
                      <td className="px-6 py-3 text-right font-mono text-text">${m.price.toFixed(2)}</td>
                      <td className={`px-6 py-3 text-right font-mono ${m.change_pct_24h >= 0 ? 'text-success' : 'text-danger'}`}>
                        {formatPercent(m.change_pct_24h)}
                      </td>
                      <td className="px-6 py-3 text-right text-text-muted">${(m.volume_24h / 1000000).toFixed(2)}M</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'positions' && (
          <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-border">
              <h3 className="text-lg font-semibold text-text">Open Positions ({positionsList.length})</h3>
            </div>
            {positionsList.length > 0 ? (
              <table className="w-full">
                <thead className="bg-bg-tertiary">
                  <tr>
                    <th className="text-left px-6 py-3 text-sm font-medium text-text-muted">Symbol</th>
                    <th className="text-left px-6 py-3 text-sm font-medium text-text-muted">Side</th>
                    <th className="text-right px-6 py-3 text-sm font-medium text-text-muted">Qty</th>
                    <th className="text-right px-6 py-3 text-sm font-medium text-text-muted">Entry</th>
                    <th className="text-right px-6 py-3 text-sm font-medium text-text-muted">Current</th>
                    <th className="text-right px-6 py-3 text-sm font-medium text-text-muted">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {positionsList.map((p: any) => {
                    const pnl = p.unrealized_pnl || 0;
                    const pnlPercent = p.entry_price > 0 ? ((p.current_price - p.entry_price) / p.entry_price) * 100 : 0;
                    return (
                      <tr key={p.position_id} className="border-t border-border">
                        <td className="px-6 py-4 font-medium text-text">{p.symbol}</td>
                        <td className="px-6 py-4">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${p.side === 'LONG' ? 'bg-success/20 text-success' : 'bg-danger/20 text-danger'}`}>
                            {p.side}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right text-text font-mono">{p.quantity.toFixed(4)}</td>
                        <td className="px-6 py-4 text-right text-text-muted font-mono">{formatCurrency(p.entry_price)}</td>
                        <td className="px-6 py-4 text-right text-text font-mono">{formatCurrency(p.current_price)}</td>
                        <td className={`px-6 py-4 text-right font-medium font-mono ${pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                          <div>{formatCurrency(pnl)}</div>
                          <div className="text-xs">{pnlPercent >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%</div>
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
    <div className="bg-bg-secondary border border-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-text-muted">{title}</span>
        <Icon className={`w-4 h-4 ${isPositive ? 'text-success' : isNegative ? 'text-danger' : 'text-text'}`} />
      </div>
      <div className={`text-xl font-bold ${isPositive ? 'text-success' : isNegative ? 'text-danger' : 'text-text'}`}>{value}</div>
      {trend !== undefined && (
        <div className={`text-xs mt-1 ${trend >= 0 ? 'text-success' : 'text-danger'}`}>
          {trend >= 0 ? '↑' : '↓'} {Math.abs(trend).toFixed(2)}%
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
    <div className="flex justify-between items-center py-2 border-b border-border/50">
      <span className="text-sm text-text-muted">{label}</span>
      <span className={`font-semibold ${valueColor}`}>{value}</span>
    </div>
  );
}

function PerfCard({ label, value, positive, inverted }: { label: string; value: string; positive?: boolean; inverted?: boolean }) {
  let valueColor = 'text-text';
  if (positive !== undefined) valueColor = positive ? 'text-success' : 'text-danger';
  if (inverted) valueColor = value.startsWith('+') ? 'text-danger' : 'text-success';
  
  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4">
      <div className="text-sm text-text-muted mb-1">{label}</div>
      <div className={`text-2xl font-bold ${valueColor}`}>{value}</div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    FILLED: 'bg-success/20 text-success',
    COMPLETED: 'bg-success/20 text-success',
    PENDING: 'bg-warning/20 text-warning',
    PARTIALLY_FILLED: 'bg-warning/20 text-warning',
    CANCELLED: 'bg-danger/20 text-danger',
    REJECTED: 'bg-danger/20 text-danger',
  };
  
  const style = styles[status] || 'bg-text-muted/20 text-text-muted';
  
  return (
    <span className={`px-2 py-1 rounded text-xs font-medium ${style}`}>
      {status}
    </span>
  );
}


