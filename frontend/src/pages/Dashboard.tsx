import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { portfolioApi, marketApi, riskApi } from '../services/api';
import { XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, Wallet, BarChart3 } from 'lucide-react';
import { DashboardSkeleton } from '../components/ui/Skeleton';
import { ErrorState } from '../components/ui/EmptyState';
import CorrelationMatrix from '../components/charts/CorrelationMatrix';
import RollingSharpeChart from '../components/charts/RollingSharpeChart';
import DrawdownChart from '../components/charts/DrawdownChart';
import MonteCarloChart from '../components/charts/MonteCarloChart';
import RiskReturnScatter from '../components/charts/RiskReturnScatter';

type TabType = 'overview' | 'performance' | 'risk' | 'market';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  
  const { data: dualSummary, isLoading: summaryLoading, error: summaryError } = useQuery({
    queryKey: ['portfolio-dual-summary'],
    queryFn: portfolioApi.getDualSummary,
    refetchInterval: 30000,
  });

  const { data: performance } = useQuery({
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
    refetchInterval: 15000,
  });

  const { data: correlationData, isLoading: correlationLoading } = useQuery({
    queryKey: ['risk-correlation'],
    queryFn: riskApi.getCorrelationMatrix,
    refetchInterval: 60000,
  });

  const { data: rollingSharpeData, isLoading: rollingSharpeLoading } = useQuery({
    queryKey: ['risk-rolling-sharpe'],
    queryFn: () => riskApi.getRollingSharpe(30),
    refetchInterval: 60000,
  });

  const { data: drawdownData, isLoading: drawdownLoading } = useQuery({
    queryKey: ['risk-drawdown'],
    queryFn: riskApi.getDrawdown,
    refetchInterval: 60000,
  });

  const { data: monteCarloData, isLoading: monteCarloLoading } = useQuery({
    queryKey: ['risk-monte-carlo'],
    queryFn: () => riskApi.getMonteCarlo(1000),
    refetchInterval: 120000,
  });

  const { data: riskReturnData, isLoading: riskReturnLoading } = useQuery({
    queryKey: ['risk-return'],
    queryFn: riskApi.getRiskReturn,
    refetchInterval: 60000,
  });

  // Use simulated as primary (paper trading)
  const realSummary = dualSummary?.simulated || dualSummary?.real;

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

  const formatCompact = (value: number) => {
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `$${(value / 1e3).toFixed(2)}K`;
    return formatCurrency(value);
  };

  const historyRows = Array.isArray(history?.history) ? history.history : [];
  const marketRows = Array.isArray(markets?.markets) ? markets.markets : [];

  const chartData = historyRows.map((entry) => ({
    date: new Date(entry.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    value: entry.value,
    return: entry.daily_return,
  }));

  if (summaryLoading && !dualSummary) {
    return <DashboardSkeleton />;
  }

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

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: BarChart3 },
    { id: 'performance' as const, label: 'Performance', icon: TrendingUp },
    { id: 'risk' as const, label: 'Risk Analysis', icon: Activity },
    { id: 'market' as const, label: 'Market', icon: DollarSign },
  ];

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-2 border-b border-border pb-4">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-all
              ${activeTab === tab.id
                ? 'bg-primary/15 text-primary border border-primary/30'
                : 'text-text-muted hover:text-text hover:bg-bg-tertiary border border-transparent'
              }
            `}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              title="Total Value"
              value={formatCurrency(realSummary?.total_value || 0)}
              icon={DollarSign}
              trend={realSummary?.daily_return_pct}
            />
            <MetricCard
              title="Daily P&L"
              value={formatCurrency(realSummary?.daily_pnl || 0)}
              icon={Activity}
              trend={realSummary?.daily_return_pct}
            />
            <MetricCard
              title="Unrealized P&L"
              value={formatCurrency(realSummary?.unrealized_pnl || 0)}
              icon={TrendingUp}
              trend={realSummary?.total_return_pct}
            />
            <MetricCard
              title="Positions"
              value={String(realSummary?.num_positions || 0)}
              icon={Wallet}
            />
          </div>

          {/* Equity Curve */}
          <div className="bg-bg-secondary border border-border rounded-xl p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-lg font-semibold text-text">Portfolio Equity</h2>
                <p className="text-sm text-text-muted">30-day portfolio value</p>
              </div>
              <div className="flex items-center gap-2 text-success">
                <TrendingUp className="w-4 h-4" />
                <span className="text-sm font-medium">{formatPercent(realSummary?.daily_return_pct || 0)}</span>
              </div>
            </div>
            {historyLoading ? (
              <div className="h-72 flex items-center justify-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              </div>
            ) : chartData.length > 0 ? (
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#58a6ff" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#58a6ff" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="date" stroke="#8b949e" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="#8b949e" fontSize={12} tickFormatter={(v) => formatCompact(v)} tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d', borderRadius: '8px' }}
                      labelStyle={{ color: '#c9d1d9' }}
                      formatter={(value: any) => [formatCurrency(Number(value) || 0), 'Value']}
                    />
                    <Area type="monotone" dataKey="value" stroke="#58a6ff" strokeWidth={2} fillOpacity={1} fill="url(#colorValue)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-72 flex items-center justify-center text-text-muted">
                No historical data available
              </div>
            )}
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <StatCard label="Total Return" value={formatPercent(performance?.total_return_pct || 0)} />
            <StatCard label="Sharpe Ratio" value={performance?.sharpe_ratio?.toFixed(2) || '0.00'} />
            <StatCard label="Win Rate" value={`${((performance?.win_rate || 0) * 100).toFixed(1)}%`} />
            <StatCard label="Max Drawdown" value={formatPercent(performance?.max_drawdown_pct || 0)} />
            <StatCard label="Sortino" value={performance?.sortino_ratio?.toFixed(2) || '0.00'} />
            <StatCard label="Calmar" value={performance?.calmar_ratio?.toFixed(2) || '0.00'} />
          </div>
        </div>
      )}

      {/* Performance Tab */}
      {activeTab === 'performance' && (
        <div className="space-y-6">
          {/* Performance Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            <PerformanceCard label="Total Return" value={formatPercent(performance?.total_return_pct || 0)} />
            <PerformanceCard label="Sharpe Ratio" value={performance?.sharpe_ratio?.toFixed(2) || '0.00'} positive={(performance?.sharpe_ratio || 0) > 1} />
            <PerformanceCard label="Sortino Ratio" value={performance?.sortino_ratio?.toFixed(2) || '0.00'} positive={(performance?.sortino_ratio || 0) > 1} />
            <PerformanceCard label="Calmar Ratio" value={performance?.calmar_ratio?.toFixed(2) || '0.00'} positive={(performance?.calmar_ratio || 0) > 1} />
            <PerformanceCard label="Win Rate" value={`${((performance?.win_rate || 0) * 100).toFixed(1)}%`} positive={(performance?.win_rate || 0) > 0.5} />
            <PerformanceCard label="Max Drawdown" value={formatPercent(performance?.max_drawdown_pct || 0)} invert />
            <PerformanceCard label="Profit Factor" value={performance?.profit_factor?.toFixed(2) || '0.00'} positive={(performance?.profit_factor || 0) > 1.5} />
          </div>

          {/* Rolling Sharpe */}
          <div className="bg-bg-secondary border border-border rounded-xl p-6">
            <h3 className="text-lg font-semibold text-text mb-2">Rolling Sharpe Ratio</h3>
            <p className="text-sm text-text-muted mb-6">30-day rolling Sharpe ratio over time</p>
            {rollingSharpeLoading ? (
              <div className="h-72 flex items-center justify-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              </div>
            ) : rollingSharpeData && rollingSharpeData.length > 0 ? (
              <div className="h-72">
                <RollingSharpeChart data={rollingSharpeData} height={280} />
              </div>
            ) : (
              <div className="h-72 flex items-center justify-center text-text-muted">
                No data available
              </div>
            )}
          </div>
        </div>
      )}

      {/* Risk Tab */}
      {activeTab === 'risk' && (
        <div className="space-y-6">
          {/* Risk Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <RiskCard label="Portfolio VaR" value="2.34%" />
            <RiskCard label="Beta" value="0.85" />
            <RiskCard label="Volatility" value="18.2%" />
            <RiskCard label="Correlation" value="0.72" />
          </div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Drawdown */}
            <div className="bg-bg-secondary border border-border rounded-xl p-6">
              <h3 className="text-lg font-semibold text-text mb-2">Drawdown Analysis</h3>
              <p className="text-sm text-text-muted mb-4">Portfolio drawdown over time</p>
              {drawdownLoading ? (
                <div className="h-72 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                </div>
              ) : drawdownData && drawdownData.length > 0 ? (
                <div className="h-72">
                  <DrawdownChart data={drawdownData} height={280} />
                </div>
              ) : (
                <div className="h-72 flex items-center justify-center text-text-muted">No data</div>
              )}
            </div>

            {/* Monte Carlo */}
            <div className="bg-bg-secondary border border-border rounded-xl p-6">
              <h3 className="text-lg font-semibold text-text mb-2">Monte Carlo Simulation</h3>
              <p className="text-sm text-text-muted mb-4">1000 simulations of potential outcomes</p>
              {monteCarloLoading ? (
                <div className="h-72 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                </div>
              ) : monteCarloData && monteCarloData.length > 0 ? (
                <div className="h-72">
                  <MonteCarloChart data={monteCarloData} height={280} />
                </div>
              ) : (
                <div className="h-72 flex items-center justify-center text-text-muted">No data</div>
              )}
            </div>
          </div>

          {/* Correlation & Risk-Return */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-bg-secondary border border-border rounded-xl p-6">
              <h3 className="text-lg font-semibold text-text mb-2">Asset Correlation</h3>
              <p className="text-sm text-text-muted mb-4">Correlation between assets</p>
              {correlationLoading ? (
                <div className="h-80 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                </div>
              ) : correlationData?.assets && correlationData?.matrix ? (
                <CorrelationMatrix data={correlationData} height={320} />
              ) : (
                <div className="h-80 flex items-center justify-center text-text-muted">No data</div>
              )}
            </div>

            <div className="bg-bg-secondary border border-border rounded-xl p-6">
              <h3 className="text-lg font-semibold text-text mb-2">Risk vs Return</h3>
              <p className="text-sm text-text-muted mb-4">Asset risk/return profile</p>
              {riskReturnLoading ? (
                <div className="h-80 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                </div>
              ) : riskReturnData && riskReturnData.length > 0 ? (
                <RiskReturnScatter data={riskReturnData} height={320} />
              ) : (
                <div className="h-80 flex items-center justify-center text-text-muted">No data</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Market Tab */}
      {activeTab === 'market' && (
        <div className="space-y-6">
          {/* Market Table */}
          <div className="bg-bg-secondary border border-border rounded-xl overflow-hidden">
            <div className="p-4 border-b border-border">
              <h2 className="text-lg font-semibold text-text">Live Market Prices</h2>
              <p className="text-sm text-text-muted">Real-time cryptocurrency prices</p>
            </div>
            {marketsLoading ? (
              <div className="h-96 flex items-center justify-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              </div>
            ) : marketRows.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-bg-tertiary/50">
                      <th className="text-left py-3 px-4 text-text-muted font-medium text-sm">Symbol</th>
                      <th className="text-right py-3 px-4 text-text-muted font-medium text-sm">Price</th>
                      <th className="text-right py-3 px-4 text-text-muted font-medium text-sm">24h Change</th>
                      <th className="text-right py-3 px-4 text-text-muted font-medium text-sm hidden sm:table-cell">24h High</th>
                      <th className="text-right py-3 px-4 text-text-muted font-medium text-sm hidden sm:table-cell">24h Low</th>
                      <th className="text-right py-3 px-4 text-text-muted font-medium text-sm hidden md:table-cell">Volume</th>
                    </tr>
                  </thead>
                  <tbody>
                    {marketRows.map((market) => (
                      <tr key={market.symbol} className="border-b border-border/50 hover:bg-bg-tertiary/30 transition-colors">
                        <td className="py-3 px-4">
                          <div className="font-medium text-text">{market.symbol.replace('USDT', '')}</div>
                          <div className="text-xs text-text-muted">USDT</div>
                        </td>
                        <td className="py-3 px-4 text-right text-text font-mono">{formatCurrency(market.price)}</td>
                        <td className={`py-3 px-4 text-right font-mono ${market.change_pct_24h >= 0 ? 'text-success' : 'text-danger'}`}>
                          {formatPercent(market.change_pct_24h)}
                        </td>
                        <td className="py-3 px-4 text-right text-text-muted font-mono hidden sm:table-cell">{formatCurrency(market.high_24h)}</td>
                        <td className="py-3 px-4 text-right text-text-muted font-mono hidden sm:table-cell">{formatCurrency(market.low_24h)}</td>
                        <td className="py-3 px-4 text-right text-text-muted font-mono hidden md:table-cell">{market.volume_24h.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="h-96 flex items-center justify-center text-text-muted">
                No market data available
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({ title, value, icon: Icon, trend }: { title: string; value: string; icon: React.ElementType; trend?: number }) {
  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4 hover:border-border-hover transition-colors">
      <div className="flex items-center justify-between mb-2">
        <span className="text-text-muted text-sm">{title}</span>
        <Icon className="w-4 h-4 text-primary" />
      </div>
      <div className="flex items-end justify-between">
        <span className="text-xl font-bold text-text">{value}</span>
        {trend !== undefined && (
          <span className={`text-sm flex items-center gap-1 ${trend >= 0 ? 'text-success' : 'text-danger'}`}>
            {trend >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            {Math.abs(trend).toFixed(2)}%
          </span>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-3">
      <div className="text-text-muted text-xs mb-1">{label}</div>
      <div className="text-text font-semibold">{value}</div>
    </div>
  );
}

function PerformanceCard({ label, value, positive, invert }: { label: string; value: string; positive?: boolean; invert?: boolean }) {
  let valueColor = 'text-text';
  if (positive !== undefined) {
    valueColor = positive ? 'text-success' : invert ? 'text-danger' : 'text-text';
  }
  
  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4">
      <div className="text-text-muted text-sm mb-1">{label}</div>
      <div className={`text-2xl font-bold ${valueColor}`}>{value}</div>
    </div>
  );
}

function RiskCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-bg-secondary border border-border rounded-xl p-4">
      <div className="text-text-muted text-sm mb-1">{label}</div>
      <div className="text-xl font-bold text-warning">{value}</div>
    </div>
  );
}
