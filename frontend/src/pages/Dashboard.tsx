import { useQuery } from '@tanstack/react-query';
import { portfolioApi, marketApi, ordersApi } from '../services/api';
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Wallet, CheckCircle, Clock, XCircle } from 'lucide-react';

export default function Dashboard() {
  // Portfolio Summary
  const { data: dualSummary, isLoading: summaryLoading } = useQuery({
    queryKey: ['portfolio-dual-summary'],
    queryFn: portfolioApi.getDualSummary,
    refetchInterval: 30000,
  });

  // Portfolio History
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

  const summary = dualSummary?.simulated || { total_value: 0, daily_pnl: 0, unrealized_pnl: 0, num_positions: 0, daily_return_pct: 0 };
  const historyData = history?.history?.map((h: any) => ({
    date: new Date(h.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    value: h.value,
  })) || [];

  const marketData = markets?.markets?.slice(0, 8) || [];
  const ordersList = (Array.isArray(orders) ? orders : []).slice(0, 10);
  const positionsList = Array.isArray(positions) ? positions : [];

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;

  if (summaryLoading && !dualSummary) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-4">
      {/* Header */}
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-text">Trading Dashboard</h1>
        <p className="text-text-muted">Paper Trading • Live Data</p>
      </div>

      {/* KEY METRICS */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card title="Total Value" value={formatCurrency(summary?.total_value || 0)} icon={DollarSign} />
        <Card 
          title="Daily P&L" 
          value={formatCurrency(summary?.daily_pnl || 0)} 
          icon={summary?.daily_pnl >= 0 ? TrendingUp : TrendingDown} 
          color={summary?.daily_pnl >= 0 ? 'text-success' : 'text-danger'}
          trend={summary?.daily_return_pct}
        />
        <Card 
          title="Unrealized P&L" 
          value={formatCurrency(summary?.unrealized_pnl || 0)} 
          icon={summary?.unrealized_pnl >= 0 ? TrendingUp : TrendingDown} 
          color={summary?.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}
        />
        <Card title="Positions" value={String(summary?.num_positions || 0)} icon={Wallet} />
      </div>

      {/* EQUITY CURVE + PERFORMANCE */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
                <XAxis dataKey="date" stroke="#8b949e" fontSize={11} interval={0} />
                <YAxis stroke="#8b949e" fontSize={11} tickFormatter={(v: number) => `$${(v/1000).toFixed(0)}k`} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '8px' }}
                  formatter={(value: any) => [formatCurrency(Number(value) || 0), 'Value']}
                />
                <Area type="monotone" dataKey="value" stroke="#3fb950" strokeWidth={2} fillOpacity={1} fill="url(#colorValue)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-bg-secondary border border-border rounded-xl p-4">
          <h2 className="text-lg font-semibold text-text mb-4">Performance</h2>
          <div className="space-y-3">
            <Stat label="Total Return" value={formatPercent(performance?.total_return_pct || 0)} color={(performance?.total_return_pct || 0) >= 0 ? 'text-success' : 'text-danger'} />
            <Stat label="Sharpe Ratio" value={String(performance?.sharpe_ratio?.toFixed(2) || '0.00')} />
            <Stat label="Win Rate" value={`${((performance?.win_rate || 0) * 100).toFixed(1)}%`} />
            <Stat label="Max Drawdown" value={formatPercent(performance?.max_drawdown_pct || 0)} color="text-danger" />
            <Stat label="Profit Factor" value={String(performance?.profit_factor?.toFixed(2) || '0.00')} />
          </div>
        </div>
      </div>

      {/* MARKET + POSITIONS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border">
            <h2 className="text-lg font-semibold text-text">Live Market Prices</h2>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-bg-tertiary/50">
                <th className="text-left py-2 px-4 text-text-muted">Symbol</th>
                <th className="text-right py-2 px-4 text-text-muted">Price</th>
                <th className="text-right py-2 px-4 text-text-muted">24h %</th>
              </tr>
            </thead>
            <tbody>
              {marketData.map((m: any) => (
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

        <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border">
            <h2 className="text-lg font-semibold text-text">Open Positions</h2>
          </div>
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
                {positionsList.map((p: any) => (
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

      {/* RECENT ORDERS */}
      <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
        <div className="p-4 border-b border-border">
          <h2 className="text-lg font-semibold text-text">Recent Orders</h2>
        </div>
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
              {ordersList.map((o: any) => (
                <tr key={o.order_id} className="border-b border-border/30">
                  <td className="py-2 px-4 text-text-muted">
                    {new Date(o.created_at).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                  </td>
                  <td className="py-2 px-4 font-medium text-text">{o.symbol}</td>
                  <td className={`py-2 px-4 ${o.side === 'BUY' ? 'text-success' : 'text-danger'}`}>{o.side}</td>
                  <td className="py-2 px-4 text-right text-text">{o.quantity.toFixed(4)}</td>
                  <td className="py-2 px-4 text-center">
                    <div className="flex items-center justify-center gap-2">
                      {o.status === 'FILLED' || o.status === 'COMPLETED' ? <CheckCircle className="w-4 h-4 text-success" /> :
                       o.status === 'PENDING' || o.status === 'PARTIALLY_FILLED' ? <Clock className="w-4 h-4 text-warning" /> :
                       <XCircle className="w-4 h-4 text-danger" />}
                      <span className={`text-xs ${o.status === 'FILLED' ? 'text-success' : o.status === 'PENDING' ? 'text-warning' : 'text-danger'}`}>
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
  );
}

function Card({ title, value, icon: Icon, color, trend }: { title: string; value: string; icon: any; color?: string; trend?: number }) {
  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-text-muted text-sm">{title}</span>
        <Icon className={`w-4 h-4 ${color || 'text-text'}`} />
      </div>
      <div className={`text-xl font-bold ${color || 'text-text'}`}>{value}</div>
      {trend !== undefined && (
        <div className={`text-xs mt-1 ${trend >= 0 ? 'text-success' : 'text-danger'}`}>
          {trend >= 0 ? '↑' : '↓'} {Math.abs(trend).toFixed(2)}%
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-text-muted text-sm">{label}</span>
      <span className={`font-semibold ${color || 'text-text'}`}>{value}</span>
    </div>
  );
}
