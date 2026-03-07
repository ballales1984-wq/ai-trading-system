import { useQuery } from '@tanstack/react-query';
import { portfolioApi, marketApi, ordersApi } from '../services/api';
import { XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, CartesianGrid, Cell } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Wallet, Activity, Clock, CheckCircle, XCircle } from 'lucide-react';

export default function Dashboard() {
  // Portfolio Summary
  const { data: dualSummary, isLoading: summaryLoading } = useQuery({
    queryKey: ['portfolio-dual-summary'],
    queryFn: portfolioApi.getDualSummary,
    refetchInterval: 30000,
  });

  // Portfolio History (Equity Curve)
  const { data: history } = useQuery({
    queryKey: ['portfolio-history'],
    queryFn: () => portfolioApi.getHistory(30),
    refetchInterval: 60000,
  });

  // Market Prices
  const { data: markets } = useQuery({
    queryKey: ['market-prices'],
    queryFn: marketApi.getAllPrices,
    refetchInterval: 15000,
  });

  // Performance
  const { data: performance } = useQuery({
    queryKey: ['portfolio-performance'],
    queryFn: portfolioApi.getPerformance,
    refetchInterval: 60000,
  });

  // Orders
  const { data: orders } = useQuery({
    queryKey: ['orders'],
    queryFn: () => ordersApi.list(),
    refetchInterval: 10000,
  });

  // Positions
  const { data: positions } = useQuery({
    queryKey: ['portfolio-positions'],
    queryFn: () => portfolioApi.getPositions(),
    refetchInterval: 30000,
  });

  // Use simulated as primary (paper trading)
  const summary = dualSummary?.simulated;
  const historyData = history?.history?.map((h) => ({
    date: new Date(h.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    value: h.value,
  })) || [];

  const marketData = markets?.markets?.slice(0, 8) || [];
  const ordersList = Array.isArray(orders) ? orders.slice(0, 10) : [];
  const positionsList = Array.isArray(positions) ? positions : [];

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'FILLED':
      case 'COMPLETED':
        return <CheckCircle className="w-4 h-4 text-success" />;
      case 'PENDING':
      case 'PARTIALLY_FILLED':
        return <Clock className="w-4 h-4 text-warning" />;
      case 'CANCELLED':
      case 'REJECTED':
        return <XCircle className="w-4 h-4 text-danger" />;
      default:
        return <Clock className="w-4 h-4 text-text-muted" />;
    }
  };

  if (summaryLoading && !dualSummary) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text">Trading Dashboard</h1>
        <p className="text-text-muted">Paper Trading Mode • Live Data</p>
      </div>

      {/* KEY METRICS - Top Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          title="Total Value"
          value={formatCurrency(summary?.total_value || 0)}
          icon={DollarSign}
          trend={summary?.daily_return_pct}
        />
        <MetricCard
          title="Daily P&L"
          value={formatCurrency(summary?.daily_pnl || 0)}
          icon={summary?.daily_pnl >= 0 ? TrendingUp : TrendingDown}
          trend={summary?.daily_return_pct}
          color={summary?.daily_pnl >= 0 ? 'success' : 'danger'}
        />
        <MetricCard
          title="Unrealized P&L"
          value={formatCurrency(summary?.unrealized_pnl || 0)}
          icon={summary?.unrealized_pnl >= 0 ? TrendingUp : TrendingDown}
          color={summary?.unrealized_pnl >= 0 ? 'success' : 'danger'}
        />
        <MetricCard
          title="Positions"
          value={String(summary?.num_positions || 0)}
          icon={Wallet}
        />
      </div>

      {/* SECOND ROW: Equity Curve + Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Equity Curve */}
        <div className="lg:col-span-2 bg-bg-secondary border border-border rounded-xl p-4">
          <h2 className="text-lg font-semibold text-text mb-4">Portfolio Equity (30 Days)</h2>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={historyData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3fb950" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3fb950" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                <XAxis dataKey="date" stroke="#8b949e" fontSize={11} />
                <YAxis stroke="#8b949e" fontSize={11} tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '8px' }}
                  formatter={(value: number) => [formatCurrency(value), 'Value']}
                />
                <Area type="monotone" dataKey="value" stroke="#3fb950" strokeWidth={2} fillOpacity={1} fill="url(#colorValue)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <h2 className="text-lg font-semibold text-text mb-4">Performance</h2>
          <div className="space-y-3">
            <StatRow label="Total Return" value={formatPercent(performance?.total_return_pct || 0)} color={(performance?.total_return_pct || 0) >= 0 ? 'success' : 'danger'} />
            <StatRow label="Sharpe Ratio" value={performance?.sharpe_ratio?.toFixed(2) || '0.00'} />
            <StatRow label="Win Rate" value={`${((performance?.win_rate || 0) * 100).toFixed(1)}%`} />
            <StatRow label="Max Drawdown" value={formatPercent(performance?.max_drawdown_pct || 0)} color="danger" />
            <StatRow label="Profit Factor" value={performance?.profit_factor?.toFixed(2) || '0.00'} />
          </div>
        </div>
      </div>

      {/* THIRD ROW: Market Prices + Positions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Live Market Prices */}
        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border">
            <h2 className="text-lg font-semibold text-text">Live Market Prices</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-bg-tertiary/50">
                  <th className="text-left py-2 px-4 text-text-muted">Symbol</th>
                  <th className="text-right py-2 px-4 text-text-muted">Price</th>
                  <th className="text-right py-2 px-4 text-text-muted">24h %</th>
                </tr>
              </thead>
              <tbody>
                {marketData.map((m) => (
                  <tr key={m.symbol} className="border-b border-border/30">
                    <td className="py-2 px-4 font-medium text-text">{m.symbol.replace('USDT', '')}</td>
                    <td className="py-2 px-4 text-right text-text font-mono">${m.price.toFixed(2)}</td>
                    <td className={`py-2 px-4 text-right font-mono ${m.change_pct_24h >= 0 ? 'text-success' : 'text-danger'}`}>
                      {formatPercent(m.change_pct_24h)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Open Positions */}
        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border">
            <h2 className="text-lg font-semibold text-text">Open Positions</h2>
          </div>
          <div className="overflow-x-auto">
            {positionsList.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-bg-tertiary/50">
                    <th className="text-left py-2 px-4 text-text-muted">Symbol</th>
                    <th className="text-left py-2 px-4 text-text-muted">Side</th>
                    <th className="text-right py-2 px-4 text-text-muted">Qty</th>
                    <th className="text-right py-2 px-4 text-text-muted">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {positionsList.map((p) => (
                    <tr key={p.position_id} className="border-b border-border/30">
                      <td className="py-2 px-4 font-medium text-text">{p.symbol}</td>
                      <td className={`py-2 px-4 ${p.side === 'LONG' ? 'text-success' : 'text-danger'}`}>{p.side}</td>
                      <td className="py-2 px-4 text-right text-text">{p.quantity.toFixed(4)}</td>
                      <td className={`py-2 px-4 text-right font-medium ${p.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                        {formatCurrency(p.unrealized_pnl)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="p-8 text-center text-text-muted">No open positions</div>
            )}
          </div>
        </div>
      </div>

      {/* FOURTH ROW: Recent Orders */}
      <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold text-text">Recent Orders</h2>
        </div>
        <div className="overflow-x-auto">
          {ordersList.length > 0 ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-bg-tertiary/50">
                  <th className="text-left py-2 px-4 text-text-muted">Time</th>
                  <th className="text-left py-2 px-4 text-text-muted">Symbol</th>
                  <th className="text-left py-2 px-4 text-text-muted">Side</th>
                  <th className="text-right py-2 px-4 text-text-muted">Qty</th>
                  <th className="text-center py-2 px-4 text-text-muted">Status</th>
                </tr>
              </thead>
              <tbody>
                {ordersList.map((o) => (
                  <tr key={o.order_id} className="border-b border-border/30">
                    <td className="py-2 px-4 text-text-muted">
                      {new Date(o.created_at).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                    </td>
                    <td className="py-2 px-4 font-medium text-text">{o.symbol}</td>
                    <td className={`py-2 px-4 ${o.side === 'BUY' ? 'text-success' : 'text-danger'}`}>{o.side}</td>
                    <td className="py-2 px-4 text-right text-text">{o.quantity.toFixed(4)}</td>
                    <td className="py-2 px-4 text-center">
                      <div className="flex items-center justify-center gap-2">
                        {getStatusIcon(o.status)}
                        <span className={`text-xs ${
                          o.status === 'FILLED' ? 'text-success' :
                          o.status === 'PENDING' ? 'text-warning' : 'text-danger'
                        }`}>
                          {o.status}
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="p-8 text-center text-text-muted">No orders yet</div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, icon: Icon, trend, color }: { title: string; value: string; icon: React.ElementType; trend?: number; color?: string }) {
  const colorClass = color === 'success' ? 'text-success' : color === 'danger' ? 'text-danger' : 'text-text';
  
  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-text-muted text-sm">{title}</span>
        <Icon className={`w-4 h-4 ${colorClass}`} />
      </div>
      <div className={`text-xl font-bold ${colorClass}`}>{value}</div>
      {trend !== undefined && (
        <div className={`text-xs mt-1 ${trend >= 0 ? 'text-success' : 'text-danger'}`}>
          {trend >= 0 ? '↑' : '↓'} {Math.abs(trend).toFixed(2)}%
        </div>
      )}
    </div>
  );
}

function StatRow({ label, value, color }: { label: string; value: string; color?: string }) {
  const colorClass = color === 'success' ? 'text-success' : color === 'danger' ? 'text-danger' : 'text-text';
  return (
    <div className="flex justify-between items-center">
      <span className="text-text-muted text-sm">{label}</span>
      <span className={`font-semibold ${colorClass}`}>{value}</span>
    </div>
  );
}
