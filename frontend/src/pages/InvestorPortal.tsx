/**
 * Investor Portal Page
 * 
 * Replaces the Dash dashboard on port 8051
 * Displays investor-focused metrics and reporting
 */

import { useQuery } from '@tanstack/react-query';
import { portfolioApi, riskApi } from '../services/api';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import { 
  Wallet, TrendingUp, TrendingDown, Shield, 
  PieChart as PieChartIcon, Calendar, Download, FileText 
} from 'lucide-react';
import { useState } from 'react';

const COLORS = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7', '#f0883e'];

interface InvestorReport {
  period: string;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
}

const mockReports: InvestorReport[] = [
  { period: 'March 2026', totalReturn: 23.5, sharpeRatio: 1.95, maxDrawdown: 7.2, winRate: 68, totalTrades: 47 },
  { period: 'February 2026', totalReturn: 18.2, sharpeRatio: 1.72, maxDrawdown: 9.5, winRate: 65, totalTrades: 52 },
  { period: 'January 2026', totalReturn: 15.8, sharpeRatio: 1.58, maxDrawdown: 11.2, winRate: 62, totalTrades: 45 },
  { period: 'December 2025', totalReturn: 21.3, sharpeRatio: 1.88, maxDrawdown: 8.1, winRate: 67, totalTrades: 49 },
];

export default function InvestorPortal() {
  const [reportPeriod, setReportPeriod] = useState<string>('March 2026');

  const { data: summary } = useQuery({
    queryKey: ['portfolio-summary'],
    queryFn: portfolioApi.getSummary,
    refetchInterval: 30000,
  });

  const { data: performance } = useQuery({
    queryKey: ['portfolio-performance'],
    queryFn: portfolioApi.getPerformance,
    refetchInterval: 60000,
  });

  const { data: allocation } = useQuery({
    queryKey: ['portfolio-allocation'],
    queryFn: portfolioApi.getAllocation,
    refetchInterval: 60000,
  });

  const { data: history } = useQuery({
    queryKey: ['portfolio-history'],
    queryFn: () => portfolioApi.getHistory(90),
    refetchInterval: 60000,
  });

  const { data: riskMetrics } = useQuery({
    queryKey: ['risk-metrics'],
    queryFn: riskApi.getMetrics,
    refetchInterval: 60000,
  });

  // Safe data transformation with fallback to mock data
  const historyData = (history?.history || []).map((h: any) => ({
    date: new Date(h.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    value: typeof h.value === 'number' ? h.value : h.total_value || h.value || 0,
  }));

  // Handle allocation data - API returns by_symbol array or nested object
  let allocationData: Array<{name: string, value: number}> = [];
  if (allocation) {
    if (Array.isArray(allocation)) {
      allocationData = allocation;
    } else if (Array.isArray((allocation as any).by_symbol)) {
      allocationData = (allocation as any).by_symbol.map((item: any) => ({
        name: item.symbol || item.name || 'Unknown',
        value: typeof item.value === 'number' ? item.value : item.allocationPct || 0,
      }));
    } else if (typeof allocation === 'object') {
      // Handle nested objects like { crypto: {...}, defi: {...} }
      allocationData = Object.entries(allocation).map(([name, value]: [string, any]) => ({
        name,
        value: typeof value === 'number' ? value : value.value || value.allocationPct || 0,
      }));
    }
  }

  // If no allocation data, use mock data
  if (allocationData.length === 0) {
    allocationData = [
      { name: 'BTC', value: 45 },
      { name: 'ETH', value: 30 },
      { name: 'SOL', value: 15 },
      { name: 'Other', value: 10 },
    ];
  }

  const currentReport = mockReports.find(r => r.period === reportPeriod) || mockReports[0];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <FileText className="w-8 h-8 text-primary" />
          <div>
            <h1 className="text-2xl font-bold text-text">Investor Portal</h1>
            <p className="text-text-muted">Performance reporting and investor relations</p>
          </div>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/80">
          <Download className="w-4 h-4" />
          Export Report
        </button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="premium-glass-panel p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-text-muted">Total AUM</span>
            <Wallet className="w-5 h-5 text-primary" />
          </div>
          <div className="text-3xl font-bold text-text">
            ${typeof (summary as any)?.totalValue === 'number' ? (summary as any).totalValue.toLocaleString() : 
              typeof (summary as any)?.total_value === 'number' ? (summary as any).total_value.toLocaleString() : '0'}
          </div>
          <div className="text-sm text-text-muted mt-1">Assets Under Management</div>
        </div>

        <div className="premium-glass-panel p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-text-muted">Total Return</span>
            {(performance?.total_return_pct ?? 0) >= 0 ? (
              <TrendingUp className="w-5 h-5 text-success" />
            ) : (
              <TrendingDown className="w-5 h-5 text-danger" />
            )}
          </div>
          <div className={`text-3xl font-bold ${((performance as any)?.total_return_pct || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
            {((performance as any)?.total_return_pct || 0) >= 0 ? '+' : ''}{((performance as any)?.total_return_pct || 0).toFixed(2)}%
          </div>
          <div className="text-sm text-text-muted mt-1">Since inception</div>
        </div>

        <div className="premium-glass-panel p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-text-muted">Sharpe Ratio</span>
            <Shield className="w-5 h-5 text-primary" />
          </div>
          <div className="text-3xl font-bold text-text">
            {typeof performance?.sharpe_ratio === 'number' ? performance.sharpe_ratio.toFixed(2) : '0.00'}
          </div>
          <div className="text-sm text-text-muted mt-1">Risk-adjusted return</div>
        </div>

        <div className="premium-glass-panel p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-text-muted">Win Rate</span>
            <PieChartIcon className="w-5 h-5 text-primary" />
          </div>
          <div className="text-3xl font-bold text-text">
            {typeof performance?.win_rate === 'number' ? performance.win_rate.toFixed(1) : '0.0'}%
          </div>
          <div className="text-sm text-text-muted mt-1">Profitable trades</div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Equity Curve */}
        <div className="lg:col-span-2 premium-glass-panel p-6">
          <h2 className="text-lg font-bold text-text mb-4">Equity Curve</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={historyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#3FB950" 
                strokeWidth={2} 
                dot={false}
                name="Portfolio Value"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Allocation */}
        <div className="premium-glass-panel p-6">
          <h2 className="text-lg font-bold text-text mb-4">Asset Allocation</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={allocationData.length > 0 ? allocationData : [
                  { name: 'BTC', value: 45 },
                  { name: 'ETH', value: 30 },
                  { name: 'SOL', value: 15 },
                  { name: 'Other', value: 10 },
                ]}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
              >
                {allocationData.length > 0 ? allocationData.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                )) : [
                  { name: 'BTC', value: 45 },
                  { name: 'ETH', value: 30 },
                  { name: 'SOL', value: 15 },
                  { name: 'Other', value: 10 },
                ].map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap justify-center gap-2 mt-2">
            {(allocationData.length > 0 ? allocationData : [
              { name: 'BTC', value: 45 },
              { name: 'ETH', value: 30 },
              { name: 'SOL', value: 15 },
              { name: 'Other', value: 10 },
            ]).map((item, index) => (
              <div key={item.name} className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
                <span className="text-xs text-text-muted">{item.name}: {item.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Period Selector */}
      <div className="flex items-center gap-2 mb-4">
        <Calendar className="w-5 h-5 text-text-muted" />
        <select 
          id="report-period"
          name="reportPeriod"
          value={reportPeriod}
          onChange={(e) => setReportPeriod(e.target.value)}
          className="bg-black/30 border border-white/10 rounded-lg px-4 py-2 text-text"
        >
          {mockReports.map(report => (
            <option key={report.period} value={report.period}>{report.period}</option>
          ))}
        </select>
      </div>

      {/* Monthly Report */}
      <div className="premium-glass-panel p-6">
        <h2 className="text-lg font-bold text-text mb-4">Monthly Performance Report</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="bg-black/20 rounded-lg p-4">
            <div className="text-sm text-text-muted mb-1">Return</div>
            <div className={`text-2xl font-bold ${currentReport.totalReturn >= 0 ? 'text-success' : 'text-danger'}`}>
              {currentReport.totalReturn >= 0 ? '+' : ''}{currentReport.totalReturn}%
            </div>
          </div>
          <div className="bg-black/20 rounded-lg p-4">
            <div className="text-sm text-text-muted mb-1">Sharpe</div>
            <div className="text-2xl font-bold text-text">{currentReport.sharpeRatio}</div>
          </div>
          <div className="bg-black/20 rounded-lg p-4">
            <div className="text-sm text-text-muted mb-1">Max DD</div>
            <div className="text-2xl font-bold text-danger">-{currentReport.maxDrawdown}%</div>
          </div>
          <div className="bg-black/20 rounded-lg p-4">
            <div className="text-sm text-text-muted mb-1">Win Rate</div>
            <div className="text-2xl font-bold text-text">{currentReport.winRate}%</div>
          </div>
          <div className="bg-black/20 rounded-lg p-4">
            <div className="text-sm text-text-muted mb-1">Trades</div>
            <div className="text-2xl font-bold text-text">{currentReport.totalTrades}</div>
          </div>
        </div>
      </div>

      {/* Risk Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="premium-glass-panel p-5">
          <h3 className="text-sm font-medium text-text-muted mb-2">Value at Risk (95%)</h3>
          <div className="text-xl font-bold text-text">
            ${(riskMetrics?.var_1d || 0).toLocaleString()}
          </div>
        </div>
        <div className="premium-glass-panel p-5">
          <h3 className="text-sm font-medium text-text-muted mb-2">Volatility (Ann.)</h3>
          <div className="text-xl font-bold text-text">
            {((riskMetrics?.volatility || 0) * 100).toFixed(1)}%
          </div>
        </div>
        <div className="premium-glass-panel p-5">
          <h3 className="text-sm font-medium text-text-muted mb-2">Beta to BTC</h3>
          <div className="text-xl font-bold text-text">
            {(riskMetrics?.correlation_to_btc || 0).toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
}
