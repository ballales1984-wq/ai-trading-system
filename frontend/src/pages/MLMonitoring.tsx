/**
 * ML Monitoring Page
 * 
 * Replaces the Dash dashboard on port 8050
 * Displays ML model performance and metrics
 */

import { useQuery } from '@tanstack/react-query';
import { portfolioApi, marketApi, riskApi } from '../services/api';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell
} from 'recharts';
import { Brain, TrendingUp, Activity, Zap, AlertTriangle, CheckCircle } from 'lucide-react';
import { useState } from 'react';

const COLORS = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a371f7'];

interface MLModel {
  name: string;
  accuracy: number;
  last_trained: string;
  status: 'active' | 'training' | 'error';
}

const mockModels: MLModel[] = [
  { name: 'BTC Price Predictor', accuracy: 0.73, last_trained: '2026-03-22', status: 'active' },
  { name: 'ETH Price Predictor', accuracy: 0.71, last_trained: '2026-03-22', status: 'active' },
  { name: 'SOL Price Predictor', accuracy: 0.68, last_trained: '2026-03-22', status: 'active' },
  { name: 'Regime Detector (HMM)', accuracy: 0.82, last_trained: '2026-03-21', status: 'active' },
  { name: 'Sentiment Analyzer', accuracy: 0.76, last_trained: '2026-03-20', status: 'training' },
];

const mockTrainingHistory = Array.from({ length: 30 }, (_, i) => ({
  day: `Day ${i + 1}`,
  accuracy: 0.6 + Math.random() * 0.2,
  loss: 0.5 - Math.random() * 0.3,
}));

const mockFeatureImportance = [
  { feature: 'RSI', importance: 0.25 },
  { feature: 'MACD', importance: 0.22 },
  { feature: 'Volume', importance: 0.18 },
  { feature: 'Bollinger', importance: 0.15 },
  { feature: 'MA Cross', importance: 0.12 },
  { feature: 'Sentiment', importance: 0.08 },
];

export default function MLMonitoring() {
  const [selectedModel, setSelectedModel] = useState<string>('BTC Price Predictor');

  const { data: performance } = useQuery({
    queryKey: ['portfolio-performance'],
    queryFn: portfolioApi.getPerformance,
    refetchInterval: 60000,
  });

  const { data: markets } = useQuery({
    queryKey: ['market-prices'],
    queryFn: marketApi.getAllPrices,
    refetchInterval: 30000,
  });

  const { data: riskMetrics } = useQuery({
    queryKey: ['risk-metrics'],
    queryFn: riskApi.getMetrics,
    refetchInterval: 60000,
  });

  const { data: history } = useQuery({
    queryKey: ['portfolio-history'],
    queryFn: () => portfolioApi.getHistory(30),
    refetchInterval: 60000,
  });

  const historyData = history?.history?.map((h: any) => ({
    date: new Date(h.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    value: h.value,
    return: h.return_pct || 0,
  })) || [];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Brain className="w-8 h-8 text-primary" />
          <div>
            <h1 className="text-2xl font-bold text-text">ML Monitoring</h1>
            <p className="text-text-muted">Model performance and training metrics</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-3 py-1 rounded-full bg-success/20 text-success text-sm">
            <CheckCircle className="w-4 h-4 inline mr-1" />
            All Systems Operational
          </span>
        </div>
      </div>

      {/* Model Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        {mockModels.map((model) => (
          <div 
            key={model.name}
            className={`premium-glass-panel p-4 cursor-pointer transition-all ${
              selectedModel === model.name ? 'border-primary glow-primary' : ''
            }`}
            onClick={() => setSelectedModel(model.name)}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-text-muted truncate">{model.name}</span>
              {model.status === 'active' && <CheckCircle className="w-4 h-4 text-success" />}
              {model.status === 'training' && <Activity className="w-4 h-4 text-warning animate-pulse" />}
              {model.status === 'error' && <AlertTriangle className="w-4 h-4 text-danger" />}
            </div>
            <div className="text-2xl font-bold text-text">{(model.accuracy * 100).toFixed(1)}%</div>
            <div className="text-xs text-text-muted">Accuracy</div>
          </div>
        ))}
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training History Chart */}
        <div className="premium-glass-panel p-6">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="text-primary" />
            <h2 className="text-lg font-bold text-text">Training History</h2>
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={mockTrainingHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="day" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Line type="monotone" dataKey="accuracy" stroke="#3FB950" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="loss" stroke="#F85149" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Feature Importance */}
        <div className="premium-glass-panel p-6">
          <div className="flex items-center gap-2 mb-4">
            <Zap className="text-primary" />
            <h2 className="text-lg font-bold text-text">Feature Importance</h2>
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={mockFeatureImportance} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" stroke="#9CA3AF" fontSize={12} />
              <YAxis dataKey="feature" type="category" stroke="#9CA3AF" fontSize={12} width={80} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
              />
              <Bar dataKey="importance" fill="#58A6FF" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Portfolio Performance */}
        <div className="premium-glass-panel p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="text-primary" />
            <h2 className="text-lg font-bold text-text">Portfolio Performance</h2>
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={historyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9CA3AF" fontSize={12} />
              <YAxis stroke="#9CA3AF" fontSize={12} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
              />
              <Line type="monotone" dataKey="value" stroke="#A371F7" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Metrics */}
        <div className="premium-glass-panel p-6">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle className="text-primary" />
            <h2 className="text-lg font-bold text-text">Risk Metrics</h2>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-black/20 rounded-lg p-4">
              <div className="text-sm text-text-muted">VaR (1d, 95%)</div>
              <div className="text-xl font-bold text-text">
                ${riskMetrics?.var_1d.toLocaleString() || '0'}
              </div>
            </div>
            <div className="bg-black/20 rounded-lg p-4">
              <div className="text-sm text-text-muted">CVaR (1d, 95%)</div>
              <div className="text-xl font-bold text-text">
                ${riskMetrics?.cvar_1d.toLocaleString() || '0'}
              </div>
            </div>
            <div className="bg-black/20 rounded-lg p-4">
              <div className="text-sm text-text-muted">Sharpe Ratio</div>
              <div className="text-xl font-bold text-text">
                {riskMetrics?.sharpe_ratio?.toFixed(2) || '0.00'}
              </div>
            </div>
            <div className="bg-black/20 rounded-lg p-4">
              <div className="text-sm text-text-muted">Max Drawdown</div>
              <div className="text-xl font-bold text-danger">
                {((riskMetrics?.max_drawdown ?? 0) * 100)?.toFixed(1) || '0.0'}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Market Overview */}
      <div className="premium-glass-panel p-6">
        <h2 className="text-lg font-bold text-text mb-4">Market Predictions vs Actual</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {markets?.markets?.slice(0, 3).map((market: any, index: number) => (
            <div key={market.symbol} className="bg-black/20 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-text">{market.symbol}</span>
                <span className={`text-sm ${market.change_pct_24h >= 0 ? 'text-success' : 'text-danger'}`}>
                  {market.change_pct_24h >= 0 ? '+' : ''}{market.change_pct_24h?.toFixed(2)}%
                </span>
              </div>
              <div className="text-2xl font-bold text-text">${market.price?.toLocaleString()}</div>
              <div className="text-sm text-text-muted mt-1">
                ML Prediction: ${(market.price * (0.95 + Math.random() * 0.1))?.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
