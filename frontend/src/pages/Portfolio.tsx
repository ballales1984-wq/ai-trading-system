import { useQuery } from '@tanstack/react-query';
import { portfolioApi, riskApi } from '../services/api';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, LineChart, Line, AreaChart, Area, CartesianGrid } from 'recharts';
import { Wallet, TrendingUp, TrendingDown, Target, Shield, Activity } from 'lucide-react';


const COLORS = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#f0883e'];

export default function Portfolio() {
  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ['portfolio-summary'],
    queryFn: portfolioApi.getSummary,
  });

  const { data: positions } = useQuery({
    queryKey: ['portfolio-positions'],
    queryFn: () => portfolioApi.getPositions(),
  });

  const { data: allocation } = useQuery({
    queryKey: ['portfolio-allocation'],
    queryFn: portfolioApi.getAllocation,
  });

  const { data: performance } = useQuery({
    queryKey: ['portfolio-performance'],
    queryFn: portfolioApi.getPerformance,
  });

  const { data: history } = useQuery({
    queryKey: ['portfolio-history'],
    queryFn: () => portfolioApi.getHistory(30),
  });

  const { data: riskMetrics } = useQuery({
    queryKey: ['risk-metrics'],
    queryFn: riskApi.getMetrics,
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

  const tradeData = performance ? [
    { name: 'Winning', value: performance.num_winning_trades },
    { name: 'Losing', value: performance.num_losing_trades },
  ] : [];

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
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-text">Portfolio</h1>
        <p className="text-text-muted">Manage your positions and allocations</p>
      </div>

      {/* Summary Cards */}
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

      {/* Charts Row 1: Equity Curve & Risk Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Equity Curve */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-text">Equity Curve</h2>
            <Activity className="w-5 h-5 text-primary" />
          </div>
          <div className="h-64">
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
                  formatter={(value: number) => [`$${value.toFixed(2)}`, 'Portfolio Value']}
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

        {/* Risk Metrics */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-text">Risk Metrics (VaR/CVaR)</h2>
            <Shield className="w-5 h-5 text-danger" />
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                <XAxis dataKey="name" stroke="#8b949e" />
                <YAxis stroke="#8b949e" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                  formatter={(value: number, name: string) => {
                    if (name === 'Volatility') return [`${value.toFixed(2)}%`, name];
                    return [`$${value.toFixed(2)}`, name];
                  }}
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
                <span className="ml-2 text-text font-semibold">{riskMetrics.beta.toFixed(2)}</span>
              </div>
              <div className="bg-border/20 rounded p-2">
                <span className="text-text-muted">Leverage:</span>
                <span className="ml-2 text-text font-semibold">{riskMetrics.leverage.toFixed(1)}x</span>
              </div>
              <div className="bg-border/20 rounded p-2">
                <span className="text-text-muted">Sharpe:</span>
                <span className="ml-2 text-text font-semibold">{riskMetrics.sharpe_ratio.toFixed(2)}</span>
              </div>
              <div className="bg-border/20 rounded p-2">
                <span className="text-text-muted">Margin:</span>
                <span className="ml-2 text-text font-semibold">{(riskMetrics.margin_utilization * 100).toFixed(0)}%</span>
              </div>
            </div>
          )}

        </div>
      </div>

      {/* Charts Row 2: Allocation & Trade Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Allocation Pie Chart */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold text-text mb-4">Asset Allocation</h2>
          <div className="h-64">
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
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {pieData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, 'Allocation']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Win/Lose Chart */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold text-text mb-4">Trade Distribution</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tradeData}>
                <XAxis dataKey="name" stroke="#8b949e" />
                <YAxis stroke="#8b949e" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                />
                <Bar dataKey="value" fill="#58a6ff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>


      {/* Positions Table */}
      <div className="bg-surface border border-border rounded-lg p-4">
        <h2 className="text-lg font-semibold text-text mb-4">Open Positions</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4 text-text-muted font-medium">Symbol</th>
                <th className="text-left py-3 px-4 text-text-muted font-medium">Side</th>
                <th className="text-right py-3 px-4 text-text-muted font-medium">Quantity</th>
                <th className="text-right py-3 px-4 text-text-muted font-medium">Entry Price</th>
                <th className="text-right py-3 px-4 text-text-muted font-medium">Current Price</th>
                <th className="text-right py-3 px-4 text-text-muted font-medium">Market Value</th>
                <th className="text-right py-3 px-4 text-text-muted font-medium">P&L</th>
                <th className="text-right py-3 px-4 text-text-muted font-medium">P&L %</th>
              </tr>
            </thead>
            <tbody>
              {positionsList.map((position) => {
                const pnlPercent = ((position.current_price - position.entry_price) / position.entry_price) * 100;
                return (
                  <tr key={position.position_id} className="border-b border-border/50 hover:bg-border/20">
                    <td className="py-3 px-4 font-medium text-text">{position.symbol}</td>
                    <td className={`py-3 px-4 ${position.side === 'LONG' ? 'text-success' : 'text-danger'}`}>
                      {position.side}
                    </td>
                    <td className="py-3 px-4 text-right text-text">{position.quantity.toFixed(4)}</td>
                    <td className="py-3 px-4 text-right text-text-muted">{formatCurrency(position.entry_price)}</td>
                    <td className="py-3 px-4 text-right text-text">{formatCurrency(position.current_price)}</td>
                    <td className="py-3 px-4 text-right text-text">{formatCurrency(position.market_value)}</td>
                    <td className={`py-3 px-4 text-right ${position.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                      {formatCurrency(position.unrealized_pnl)}
                    </td>
                    <td className={`py-3 px-4 text-right ${pnlPercent >= 0 ? 'text-success' : 'text-danger'}`}>
                      {pnlPercent >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%
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

function SummaryCard({ 
  title, 
  value, 
  icon: Icon, 
  valueColor = 'text-text' 
}: { 
  title: string; 
  value: string; 
  icon: React.ElementType; 
  valueColor?: string;
}) {
  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-text-muted text-sm">{title}</span>
        <Icon className="w-5 h-5 text-primary" />
      </div>
      <div className={`text-xl font-bold ${valueColor}`}>{value}</div>
    </div>
  );
}
