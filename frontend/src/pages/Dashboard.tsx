import { useQuery } from '@tanstack/react-query';
import { portfolioApi, marketApi } from '../services/api';
import { XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, Wallet } from 'lucide-react';
import { DashboardSkeleton } from '../components/ui/Skeleton';
import { ErrorState } from '../components/ui/EmptyState';

export default function Dashboard() {
  const { data: summary, isLoading: summaryLoading, error: summaryError } = useQuery({
    queryKey: ['portfolio-summary'],
    queryFn: portfolioApi.getSummary,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: performance, isLoading: perfLoading } = useQuery({
    queryKey: ['portfolio-performance'],
    queryFn: portfolioApi.getPerformance,
  });

  const { data: history, isLoading: historyLoading } = useQuery({
    queryKey: ['portfolio-history'],
    queryFn: () => portfolioApi.getHistory(30),
  });

  const { data: markets, isLoading: marketsLoading } = useQuery({
    queryKey: ['market-prices'],
    queryFn: marketApi.getAllPrices,
    refetchInterval: 15000, // Refresh every 15 seconds
  });

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

  const chartData = history?.history.map((entry) => ({
    date: new Date(entry.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    value: entry.value,
    return: entry.daily_return,
  })) || [];

  // Show loading skeleton
  if (summaryLoading && !summary) {
    return <DashboardSkeleton />;
  }

  // Show error state
  if (summaryError) {
    return (
      <div className="p-6">
        <ErrorState
          title="Failed to load dashboard"
          message="Unable to fetch portfolio data. Please check your connection and try again."
          retry={() => window.location.reload()}
        />
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-text">Dashboard</h1>
        <p className="text-text-muted">Overview of your trading portfolio</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard
          title="Total Value"
          value={summaryLoading ? '...' : formatCurrency(summary?.total_value || 0)}
          icon={DollarSign}
          trend={summary?.daily_return_pct}
        />
        <MetricCard
          title="Daily P&L"
          value={summaryLoading ? '...' : formatCurrency(summary?.daily_pnl || 0)}
          icon={Activity}
          trend={summary?.daily_return_pct}
        />
        <MetricCard
          title="Unrealized P&L"
          value={summaryLoading ? '...' : formatCurrency(summary?.unrealized_pnl || 0)}
          icon={TrendingUp}
          trend={summary?.total_return_pct}
        />
        <MetricCard
          title="Positions"
          value={summaryLoading ? '...' : String(summary?.num_positions || 0)}
          icon={Wallet}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Equity Curve */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold text-text mb-4">Equity Curve</h2>
          {historyLoading ? (
            <div className="h-64 flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
          ) : chartData.length > 0 ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#58a6ff" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#58a6ff" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="date" stroke="#8b949e" fontSize={12} />
                  <YAxis stroke="#8b949e" fontSize={12} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                    labelStyle={{ color: '#c9d1d9' }}
                    formatter={(value: number) => [formatCurrency(value), 'Value']}
                  />
                  <Area type="monotone" dataKey="value" stroke="#58a6ff" fillOpacity={1} fill="url(#colorValue)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-text-muted">
              No historical data available
            </div>
          )}
        </div>

        {/* Performance Metrics */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold text-text mb-4">Performance Metrics</h2>
          {perfLoading ? (
            <div className="h-64 flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              <PerformanceItem label="Total Return" value={formatPercent(performance?.total_return_pct || 0)} />
              <PerformanceItem label="Sharpe Ratio" value={performance?.sharpe_ratio?.toFixed(2) || '0.00'} />
              <PerformanceItem label="Win Rate" value={`${((performance?.win_rate || 0) * 100).toFixed(1)}%`} />
              <PerformanceItem label="Max Drawdown" value={formatPercent(performance?.max_drawdown_pct || 0)} />
              <PerformanceItem label="Sortino Ratio" value={performance?.sortino_ratio?.toFixed(2) || '0.00'} />
              <PerformanceItem label="Calmar Ratio" value={performance?.calmar_ratio?.toFixed(2) || '0.00'} />
            </div>
          )}
        </div>
      </div>

      {/* Market Overview */}
      <div className="bg-surface border border-border rounded-lg p-4">
        <h2 className="text-lg font-semibold text-text mb-4">Market Overview</h2>
        {marketsLoading ? (
          <div className="h-48 flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        ) : markets?.markets && markets.markets.length > 0 ? (
          <div className="overflow-x-auto table-responsive">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-4 text-text-muted font-medium">Symbol</th>
                  <th className="text-right py-3 px-4 text-text-muted font-medium">Price</th>
                  <th className="text-right py-3 px-4 text-text-muted font-medium">24h Change</th>
                  <th className="text-right py-3 px-4 text-text-muted font-medium">24h High</th>
                  <th className="text-right py-3 px-4 text-text-muted font-medium">24h Low</th>
                  <th className="text-right py-3 px-4 text-text-muted font-medium">Volume</th>
                </tr>
              </thead>
              <tbody>
                {markets.markets.map((market) => (
                  <tr key={market.symbol} className="border-b border-border/50 hover:bg-border/20">
                    <td className="py-3 px-4 font-medium text-text">{market.symbol}</td>
                    <td className="py-3 px-4 text-right text-text">{formatCurrency(market.price)}</td>
                    <td className={`py-3 px-4 text-right ${market.change_pct_24h >= 0 ? 'text-success' : 'text-danger'}`}>
                      {formatPercent(market.change_pct_24h)}
                    </td>
                    <td className="py-3 px-4 text-right text-text-muted">{formatCurrency(market.high_24h)}</td>
                    <td className="py-3 px-4 text-right text-text-muted">{formatCurrency(market.low_24h)}</td>
                    <td className="py-3 px-4 text-right text-text-muted">{market.volume_24h.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="h-48 flex items-center justify-center text-text-muted">
            No market data available
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({ title, value, icon: Icon, trend }: { title: string; value: string; icon: React.ElementType; trend?: number }) {
  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-text-muted text-sm">{title}</span>
        <Icon className="w-5 h-5 text-primary" />
      </div>
      <div className="flex items-end justify-between">
        <span className="text-xl font-bold text-text">{value}</span>
        {trend !== undefined && (
          <span className={`text-sm ${trend >= 0 ? 'text-success' : 'text-danger'}`}>
            {trend >= 0 ? <TrendingUp className="w-4 h-4 inline" /> : <TrendingDown className="w-4 h-4 inline" />}
            {' '}{Math.abs(trend).toFixed(2)}%
          </span>
        )}
      </div>
    </div>
  );
}

function PerformanceItem({ label, value }: { label: string; value: string }) {
  const isPositive = value.startsWith('+');
  const isNegative = value.startsWith('-') && !value.includes('%');
  const valueColor = isPositive ? 'text-success' : isNegative ? 'text-danger' : 'text-text';

  return (
    <div className="p-3 bg-background rounded-lg">
      <div className="text-text-muted text-sm">{label}</div>
      <div className={`text-lg font-semibold ${valueColor}`}>{value}</div>
    </div>
  );
}
