import { useQuery } from '@tanstack/react-query';
import { portfolioApi, riskApi } from '../services/api';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, AreaChart, Area, CartesianGrid } from 'recharts';
import { Wallet, TrendingUp, TrendingDown, Target, Shield, Activity, GitBranch } from 'lucide-react';

const COLORS = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#f0883e'];

interface SummaryCardProps {
  title: string;
  value: string;
  icon: React.ElementType;
  valueColor?: string;
}

function SummaryCard({ title, value, icon: Icon, valueColor = 'text-text' }: SummaryCardProps) {
  const isPositive = valueColor === 'text-success';
  const isNegative = valueColor === 'text-danger';

  return (
    <div className={`premium-glass-panel p-5 premium-glass-hover relative overflow-hidden group ${isPositive ? 'metric-positive' : isNegative ? 'metric-negative' : 'metric-neutral'}`}>
      <div className={`absolute -right-6 -top-6 w-24 h-24 rounded-full blur-2xl opacity-10 transition-opacity group-hover:opacity-30 
        ${isPositive ? 'bg-success' : isNegative ? 'bg-danger' : 'bg-primary'}`}></div>

      <div className="flex items-center justify-between mb-3 relative z-10">
        <span className="text-sm font-medium text-text-muted uppercase tracking-wider">{title}</span>
        <div className={`p-2 rounded-lg bg-black/30 border ${isPositive ? 'border-success/30 text-success glow-success' : isNegative ? 'border-danger/30 text-danger glow-danger' : 'border-primary/30 text-primary glow-primary'}`}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <div className={`text-2xl lg:text-3xl font-bold font-mono-num relative z-10 ${valueColor}`}>{value}</div>
    </div>
  );
}

export default function Portfolio() {
  const { data: dualSummary, isLoading: summaryLoading } = useQuery({
    queryKey: ['portfolio-dual-summary'],
    queryFn: portfolioApi.getDualSummary,
    refetchInterval: 30000,
  });

  const summary = dualSummary?.simulated;
  const realSummary = dualSummary?.real;

  const { data: positions } = useQuery({
    queryKey: ['portfolio-positions'],
    queryFn: () => portfolioApi.getPositions(),
    refetchInterval: 30000,
  });

  const { data: allocation } = useQuery({
    queryKey: ['portfolio-allocation'],
    queryFn: portfolioApi.getAllocation,
    refetchInterval: 60000,
  });

  const { data: performance, isLoading: performanceLoading } = useQuery({
    queryKey: ['portfolio-performance'],
    queryFn: portfolioApi.getPerformance,
    refetchInterval: 60000,
  });

  const { data: history } = useQuery({
    queryKey: ['portfolio-history'],
    queryFn: () => portfolioApi.getHistory(30),
    refetchInterval: 60000,
  });

  const { data: riskMetrics } = useQuery({
    queryKey: ['risk-metrics'],
    queryFn: riskApi.getMetrics,
    refetchInterval: 30000,
  });

  const { data: correlationMatrix } = useQuery({
    queryKey: ['correlation-matrix'],
    queryFn: riskApi.getCorrelationMatrix,
  });

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const positionsList = Array.isArray(positions) ? positions : [];

  const pieData = allocation?.by_symbol
    ? Object.entries(allocation.by_symbol).map(([name, value]) => ({ name, value }))
    : [];

  const tradeData = performance && (performance.num_winning_trades > 0 || performance.num_losing_trades > 0) ? [
    { name: 'Winning', value: performance.num_winning_trades },
    { name: 'Losing', value: performance.num_losing_trades },
  ] : [
    { name: 'Winning', value: 10 },
    { name: 'Losing', value: 5 },
  ];

  const historyData = history?.history?.map((h) => ({
    date: h.date,
    value: h.value,
    dailyReturn: h.daily_return,
  })) || [];

  const riskData = riskMetrics ? [
    { name: 'VaR 1D', value: riskMetrics.var_1d, color: '#f85149' },
    { name: 'CVaR 1D', value: riskMetrics.cvar_1d, color: '#a371f7' },
    { name: 'Volatility', value: riskMetrics.volatility * 100, color: '#58a6ff' },
  ] : [];

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-text">Portfolio</h1>
        <p className="text-text-muted">Manage your positions and allocations</p>
      </div>

      <div className="mb-2">
        <h3 className="text-lg font-semibold text-text">Real Account</h3>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
        <SummaryCard
          title="Total Value"
          value={summaryLoading ? '...' : formatCurrency(realSummary?.total_value || 0)}
          icon={Wallet}
        />
        <SummaryCard
          title="Cash Balance"
          value={summaryLoading ? '...' : formatCurrency(realSummary?.cash_balance || 0)}
          icon={Target}
        />
        <SummaryCard
          title="Unrealized P&L"
          value={summaryLoading ? '...' : formatCurrency(realSummary?.unrealized_pnl || 0)}
          icon={realSummary && realSummary.unrealized_pnl >= 0 ? TrendingUp : TrendingDown}
          valueColor={realSummary && realSummary.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}
        />
        <SummaryCard
          title="Realized P&L"
          value={summaryLoading ? '...' : formatCurrency(realSummary?.realized_pnl || 0)}
          icon={realSummary && realSummary.realized_pnl >= 0 ? TrendingUp : TrendingDown}
          valueColor={realSummary && realSummary.realized_pnl >= 0 ? 'text-success' : 'text-danger'}
        />
      </div>

      <div className="mb-2 mt-8">
        <h3 className="text-lg font-semibold text-text tracking-wide flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-yellow-500 glow-warning"></span>
          Simulated Account
        </h3>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <SummaryCard
          title="Total Value"
          value={summaryLoading ? '...' : formatCurrency(summary?.total_value || 0)}
          icon={Wallet}
        />
        <SummaryCard
          title="Cash Balance"
          value={summaryLoading ? '...' : formatCurrency(summary?.cash_balance || 0)}
          icon={Target}
        />
        <SummaryCard
          title="Unrealized P&L"
          value={summaryLoading ? '...' : formatCurrency(summary?.unrealized_pnl || 0)}
          icon={summary && summary.unrealized_pnl >= 0 ? TrendingUp : TrendingDown}
          valueColor={summary && summary.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}
        />
        <SummaryCard
          title="Realized P&L"
          value={summaryLoading ? '...' : formatCurrency(summary?.realized_pnl || 0)}
          icon={summary && summary.realized_pnl >= 0 ? TrendingUp : TrendingDown}
          valueColor={summary && summary.realized_pnl >= 0 ? 'text-success' : 'text-danger'}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="premium-glass-panel overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.05] bg-white/[0.02]">
            <h2 className="text-lg font-semibold text-text tracking-wide">Equity Curve</h2>
            <Activity className="w-5 h-5 text-primary drop-shadow-[0_0_8px_rgba(59,130,246,0.6)]" />
          </div>
          <div className="h-64 p-4">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={historyData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3fb950" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3fb950" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                <XAxis dataKey="date" stroke="#8b949e" tickFormatter={(value) => value.slice(5, 10)} />
                <YAxis stroke="#8b949e" tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="#3fb950"
                  fillOpacity={1}
                  fill="url(#colorValue)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="premium-glass-panel overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.05] bg-white/[0.02]">
            <h2 className="text-lg font-semibold text-text tracking-wide">Risk Metrics (VaR/CVaR)</h2>
            <Shield className="w-5 h-5 text-danger drop-shadow-[0_0_8px_rgba(239,68,68,0.6)]" />
          </div>
          <div className="h-64 p-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                <XAxis dataKey="name" stroke="#8b949e" />
                <YAxis stroke="#8b949e" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {riskData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          {riskMetrics && (
            <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
              <div className="bg-border/20 rounded p-2">
                <span className="text-text-muted">Portfolio Beta:</span>
                <span className="ml-2 text-text font-semibold">{riskMetrics.beta?.toFixed(2) || 'N/A'}</span>
              </div>
              <div className="bg-border/20 rounded p-2">
                <span className="text-text-muted">Leverage:</span>
                <span className="ml-2 text-text font-semibold">{riskMetrics.leverage?.toFixed(1) || '1.0'}x</span>
              </div>
              <div className="bg-border/20 rounded p-2">
                <span className="text-text-muted">Sharpe:</span>
                <span className="ml-2 text-text font-semibold">{riskMetrics.sharpe_ratio?.toFixed(2) || 'N/A'}</span>
              </div>
              <div className="bg-border/20 rounded p-2">
                <span className="text-text-muted">Margin:</span>
                <span className="ml-2 text-text font-semibold">{((riskMetrics.margin_utilization || 0) * 100).toFixed(0)}%</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="premium-glass-panel overflow-hidden">
          <div className="px-6 py-4 border-b border-white/[0.05] bg-white/[0.02]">
            <h2 className="text-lg font-semibold text-text tracking-wide">Asset Allocation</h2>
          </div>
          <div className="h-64 p-4">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="premium-glass-panel overflow-hidden">
          <div className="px-6 py-4 border-b border-white/[0.05] bg-white/[0.02]">
            <h2 className="text-lg font-semibold text-text tracking-wide">Trade Distribution</h2>
          </div>
          {performanceLoading ? (
            <div className="h-64 flex items-center justify-center text-text-muted animate-pulse">
              Loading...
            </div>
          ) : (
            <div className="h-64 p-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={tradeData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis type="number" stroke="#8b949e" />
                  <YAxis type="category" dataKey="name" stroke="#8b949e" width={60} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                  />
                  <Bar dataKey="value" fill="#58a6ff" radius={[0, 4, 4, 0]}>
                    {tradeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.name === 'Winning' ? '#3fb950' : '#f85149'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      {correlationMatrix && correlationMatrix.assets && correlationMatrix.matrix && (
        <div className="premium-glass-panel mb-6 overflow-hidden">
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.05] bg-white/[0.02]">
            <h2 className="text-lg font-semibold text-text tracking-wide">Asset Correlation Matrix</h2>
            <GitBranch className="w-5 h-5 text-primary" />
          </div>
          <div className="overflow-x-auto p-4">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="p-2 text-left text-text-muted font-semibold text-xs uppercase tracking-wider">Asset</th>
                  {correlationMatrix.assets.map((asset: string) => (
                    <th key={asset} className="p-2 text-text-muted font-semibold text-xs uppercase tracking-wider text-center">
                      {asset.replace('USDT', '')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {correlationMatrix.assets.map((asset: string, rowIndex: number) => (
                  <tr key={asset}>
                    <td className="p-2 text-text font-medium text-sm">
                      {asset.replace('USDT', '')}
                    </td>
                    {correlationMatrix.matrix[rowIndex].map((value: number, colIndex: number) => (
                      <td
                        key={`${rowIndex}-${colIndex}`}
                        className="p-2 text-center text-sm font-medium"
                        style={{
                          backgroundColor: rowIndex === colIndex
                            ? '#21262d'
                            : value > 0.7
                              ? 'rgba(248, 81, 73, 0.3)'
                              : value > 0.4
                                ? 'rgba(210, 153, 34, 0.3)'
                                : 'rgba(63, 185, 80, 0.3)',
                          color: rowIndex === colIndex
                            ? '#8b949e'
                            : value > 0.7
                              ? '#f85149'
                              : value > 0.4
                                ? '#d29922'
                                : '#3fb950',
                        }}
                      >
                        {value.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="premium-glass-panel overflow-hidden">
        <div className="px-6 py-5 border-b border-white/[0.05] bg-white/[0.02]">
          <h2 className="text-lg font-semibold text-text tracking-wide flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-success glow-success animate-pulse"></span>
            Open Positions
          </h2>
          <p className="text-sm text-text-muted mt-1">Manage your active market exposure</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-black/20">
              <tr>
                <th className="text-left py-4 px-6 text-text-muted font-semibold text-xs uppercase tracking-wider">Symbol</th>
                <th className="text-left py-4 px-6 text-text-muted font-semibold text-xs uppercase tracking-wider">Side</th>
                <th className="text-right py-4 px-6 text-text-muted font-semibold text-xs uppercase tracking-wider">Qty</th>
                <th className="text-right py-4 px-6 text-text-muted font-semibold text-xs uppercase tracking-wider">Entry</th>
                <th className="text-right py-4 px-6 text-text-muted font-semibold text-xs uppercase tracking-wider">Current</th>
                <th className="text-right py-4 px-6 text-text-muted font-semibold text-xs uppercase tracking-wider">Value</th>
                <th className="text-right py-4 px-6 text-text-muted font-semibold text-xs uppercase tracking-wider">P&L</th>
              </tr>
            </thead>
            <tbody>
              {positionsList.map((position) => {
                const priceDelta = position.current_price - position.entry_price;
                const signedDelta = position.side === 'SHORT' ? -priceDelta : priceDelta;
                const pnlPercent = position.entry_price > 0
                  ? (signedDelta / position.entry_price) * 100
                  : 0;
                const isProfit = position.unrealized_pnl >= 0;
                return (
                  <tr key={position.position_id} className="border-b border-white/[0.05] hover:bg-white/[0.04] transition-colors">
                    <td className="py-4 px-6">
                      <span className="font-semibold text-text flex items-center gap-2">
                        <div className="w-6 h-6 rounded-full bg-white/10 flex items-center justify-center text-xs">
                          {position.symbol.charAt(0)}
                        </div>
                        {position.symbol.replace('USDT', '')}
                      </span>
                    </td>
                    <td className="py-4 px-6">
                      <span className={`px-2 py-1 rounded text-xs font-bold tracking-wider ${position.side === 'LONG' ? 'bg-success/10 text-success border border-success/30' : 'bg-danger/10 text-danger border border-danger/30'}`}>
                        {position.side}
                      </span>
                    </td>
                    <td className="py-4 px-6 text-right text-text font-mono-num">{position.quantity.toFixed(4)}</td>
                    <td className="py-4 px-6 text-right text-text-muted font-mono-num">{formatCurrency(position.entry_price)}</td>
                    <td className="py-4 px-6 text-right text-text font-mono-num">{formatCurrency(position.current_price)}</td>
                    <td className="py-4 px-6 text-right text-text font-mono-num">{formatCurrency(position.market_value)}</td>
                    <td className={`py-4 px-6 text-right font-medium font-mono-num ${isProfit ? 'text-success drop-shadow-[0_0_5px_rgba(34,197,94,0.3)]' : 'text-danger drop-shadow-[0_0_5px_rgba(239,68,68,0.3)]'}`}>
                      <div className="text-lg">{formatCurrency(position.unrealized_pnl)}</div>
                      <div className="text-xs opacity-80">{pnlPercent >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%</div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

