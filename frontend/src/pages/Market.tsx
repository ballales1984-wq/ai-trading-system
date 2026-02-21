import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { marketApi } from '../services/api';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CandlestickChart, Candlestick, BarChart, Bar } from 'recharts';
import { Search, TrendingUp, TrendingDown, Activity } from 'lucide-react';

export default function Market() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1h');

  const { data: prices } = useQuery({
    queryKey: ['market-prices'],
    queryFn: marketApi.getAllPrices,
  });

  const { data: candles } = useQuery({
    queryKey: ['market-candles', selectedSymbol, timeframe],
    queryFn: () => marketApi.getCandles(selectedSymbol, timeframe, 100),
    refetchInterval: 30000,
  });

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatPrice = (value: number) => {
    if (value >= 1000) return value.toFixed(2);
    if (value >= 1) return value.toFixed(4);
    return value.toFixed(6);
  };

  const chartData = candles?.map((candle) => ({
    timestamp: new Date(candle.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume,
  })) || [];

  const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'EURUSD'];

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-text">Market</h1>
        <p className="text-text-muted">Real-time market data and charts</p>
      </div>

      {/* Symbol Selector */}
      <div className="flex gap-4 mb-6">
        <div className="flex gap-2">
          {symbols.map((symbol) => (
            <button
              key={symbol}
              onClick={() => setSelectedSymbol(symbol)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedSymbol === symbol
                  ? 'bg-primary text-white'
                  : 'bg-surface border border-border text-text-muted hover:text-text'
              }`}
            >
              {symbol}
            </button>
          ))}
        </div>
        <div className="flex gap-2 ml-auto">
          {['15m', '1h', '4h', '1d'].map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                timeframe === tf
                  ? 'bg-primary/20 text-primary'
                  : 'bg-surface border border-border text-text-muted hover:text-text'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* Price Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {prices?.markets
          .filter((m) => m.symbol === selectedSymbol)
          .map((market) => (
            <>
              <PriceCard
                title="Current Price"
                value={formatCurrency(market.price)}
                icon={market.change_pct_24h >= 0 ? TrendingUp : TrendingDown}
              />
              <PriceCard
                title="24h Change"
                value={`${market.change_pct_24h >= 0 ? '+' : ''}${market.change_pct_24h.toFixed(2)}%`}
                valueColor={market.change_pct_24h >= 0 ? 'text-success' : 'text-danger'}
              />
              <PriceCard
                title="24h High"
                value={formatCurrency(market.high_24h)}
              />
              <PriceCard
                title="24h Low"
                value={formatCurrency(market.low_24h)}
              />
            </>
          ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Price Chart */}
        <div className="lg:col-span-2 bg-surface border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold text-text mb-4">{selectedSymbol} Price Chart</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <XAxis dataKey="timestamp" stroke="#8b949e" fontSize={10} interval="preserveStartEnd" />
                <YAxis stroke="#8b949e" fontSize={10} domain={['auto', 'auto']} tickFormatter={(v) => formatPrice(v)} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                  labelStyle={{ color: '#c9d1d9' }}
                  formatter={(value: number) => [formatCurrency(value), 'Price']}
                />
                <Line
                  type="monotone"
                  dataKey="close"
                  stroke="#58a6ff"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Volume Chart */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold text-text mb-4">Volume</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <XAxis dataKey="timestamp" stroke="#8b949e" fontSize={10} interval="preserveStartEnd" />
                <YAxis stroke="#8b949e" fontSize={10} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#161b22', border: '1px solid #30363d' }}
                  formatter={(value: number) => [value.toFixed(2), 'Volume']}
                />
                <Bar dataKey="volume" fill="#3fb950" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* All Markets Table */}
      <div className="mt-6 bg-surface border border-border rounded-lg p-4">
        <h2 className="text-lg font-semibold text-text mb-4">All Markets</h2>
        <div className="overflow-x-auto">
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
              {prices?.markets.map((market) => (
                <tr
                  key={market.symbol}
                  className="border-b border-border/50 hover:bg-border/20 cursor-pointer"
                  onClick={() => setSelectedSymbol(market.symbol)}
                >
                  <td className="py-3 px-4 font-medium text-text">{market.symbol}</td>
                  <td className="py-3 px-4 text-right text-text">{formatCurrency(market.price)}</td>
                  <td className={`py-3 px-4 text-right ${market.change_pct_24h >= 0 ? 'text-success' : 'text-danger'}`}>
                    {market.change_pct_24h >= 0 ? '+' : ''}{market.change_pct_24h.toFixed(2)}%
                  </td>
                  <td className="py-3 px-4 text-right text-text-muted">{formatCurrency(market.high_24h)}</td>
                  <td className="py-3 px-4 text-right text-text-muted">{formatCurrency(market.low_24h)}</td>
                  <td className="py-3 px-4 text-right text-text-muted">{market.volume_24h.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function PriceCard({ title, value, icon: Icon, valueColor = 'text-text' }: { title: string; value: string; icon: React.ElementType; valueColor?: string }) {
  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-text-muted text-sm">{title}</span>
        <Icon className={`w-5 h-5 ${valueColor === 'text-success' ? 'text-success' : valueColor === 'text-danger' ? 'text-danger' : 'text-primary'}`} />
      </div>
      <div className={`text-xl font-bold ${valueColor}`}>{value}</div>
    </div>
  );
}

