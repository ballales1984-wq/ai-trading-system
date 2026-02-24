import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { marketApi } from '../services/api';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';
import { DashboardSkeleton } from '../components/ui/Skeleton';
import { EmptyState, ErrorState } from '../components/ui/EmptyState';
import { formatCurrencyUSD } from '../utils/format';

// TypeScript interfaces for type safety
interface MarketData {
  symbol: string;
  price: number;
  change_pct_24h: number;
  high_24h: number;
  low_24h: number;
  volume_24h: number;
}

interface PricesResponse {
  markets: MarketData[];
}

interface CandleData {
  timestamp: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'EURUSD'];
const ALLOW_FALLBACK =
  import.meta.env.VITE_ALLOW_MARKET_FALLBACK === 'true' ||
  (import.meta.env.VITE_ALLOW_MARKET_FALLBACK === undefined && import.meta.env.DEV);

// Fallback data for when API is unavailable (deterministic for consistent renders)
const fallbackPrices: PricesResponse = {
  markets: [
    { symbol: 'BTCUSDT', price: 43500, change_pct_24h: 2.5, high_24h: 44000, low_24h: 42000, volume_24h: 5000000000 },
    { symbol: 'ETHUSDT', price: 2350, change_pct_24h: 1.8, high_24h: 2400, low_24h: 2300, volume_24h: 2000000000 },
    { symbol: 'SOLUSDT', price: 95, change_pct_24h: -0.5, high_24h: 98, low_24h: 92, volume_24h: 500000000 },
    { symbol: 'BNBUSDT', price: 310, change_pct_24h: 0.3, high_24h: 315, low_24h: 305, volume_24h: 100000000 },
    { symbol: 'EURUSD', price: 1.085, change_pct_24h: 0.1, high_24h: 1.09, low_24h: 1.08, volume_24h: 50000000 },
  ]
};

// Generate deterministic fallback candles to prevent inconsistent renders
const generateFallbackCandles = (): CandleData[] => 
  Array.from({ length: 50 }, (_, i) => ({
    timestamp: new Date(Date.now() - (50 - i) * 3600000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    open: 43000 + i * 15,
    high: 44000 + i * 15,
    low: 42000 + i * 15,
    close: 43500 + i * 15,
    volume: 500 + i * 10,
  }));

export default function Market() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1h');

  // Memoize fallback candles to prevent inconsistent renders
  const fallbackCandles = useMemo(() => generateFallbackCandles(), []);

  const { data: prices, isLoading: pricesLoading, error: pricesError } = useQuery({
    queryKey: ['market-prices'],
    queryFn: marketApi.getAllPrices,
    retry: 1,
    staleTime: 30000,
  });

  const { data: candles, error: candlesError } = useQuery({
    queryKey: ['market-candles', selectedSymbol, timeframe],
    queryFn: () => marketApi.getCandles(selectedSymbol, timeframe, 100),
    retry: 1,
    staleTime: 30000,
  });

  // Use fallback data only when real payload is missing.
  // Avoid false "demo data" warnings on transient query errors.
  const hasPricesData = !!(prices?.markets && prices.markets.length > 0);
  const hasCandlesData = Array.isArray(candles) && candles.length > 0;
  const pricesData = hasPricesData ? prices : ALLOW_FALLBACK ? fallbackPrices : { markets: [] };
  const candlesData = hasCandlesData ? candles : ALLOW_FALLBACK ? fallbackCandles : [];
  const isUsingFallback = ALLOW_FALLBACK && (!hasPricesData || !hasCandlesData);

  const formatPrice = (value: number) => {
    if (value >= 1000) return value.toFixed(2);
    if (value >= 1) return value.toFixed(4);
    return value.toFixed(6);
  };

  const chartData = candlesData?.map((candle: CandleData) => ({
    timestamp: typeof candle.timestamp === 'string' ? candle.timestamp : new Date(candle.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume,
  })) || [];

  // Get selected market data
  const selectedMarket = pricesData?.markets?.find((m: MarketData) => m.symbol === selectedSymbol);

  if (pricesLoading && !prices) {
    return <DashboardSkeleton />;
  }

  if (pricesError || candlesError) {
    return (
      <div className="p-6">
        <ErrorState
          title="Failed to load market data"
          message="Backend/API provider returned an error."
          retry={() => window.location.reload()}
        />
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-text">Market</h1>
        <p className="text-text-muted">Real-time market data and charts</p>
      </div>

      {/* Fallback Data Warning */}
      {isUsingFallback && (
        <div className="mb-6 bg-yellow-500/20 border border-yellow-500/50 rounded-lg p-4 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0" />
          <div>
            <p className="text-yellow-500 font-medium">Using Demo Data</p>
            <p className="text-text-muted text-sm">API connection unavailable. Showing simulated market data for demonstration purposes.</p>
          </div>
        </div>
      )}
      {!ALLOW_FALLBACK && (!hasPricesData || !hasCandlesData) && (
        <div className="mb-6 rounded-lg border border-danger/50 bg-danger/10 p-4 text-sm text-danger">
          Live market data unavailable. Demo fallback is disabled in this environment.
        </div>
      )}

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
      {pricesLoading || !selectedMarket ? (
        <div className="text-center text-text-muted py-8">Loading market data...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <PriceCard
            title="Current Price"
            value={formatCurrencyUSD(selectedMarket.price)}
            icon={selectedMarket.change_pct_24h >= 0 ? TrendingUp : TrendingDown}
          />
          <PriceCard
            title="24h Change"
            value={`${selectedMarket.change_pct_24h >= 0 ? '+' : ''}${selectedMarket.change_pct_24h.toFixed(2)}%`}
            valueColor={selectedMarket.change_pct_24h >= 0 ? 'text-success' : 'text-danger'}
          />
          <PriceCard
            title="24h High"
            value={formatCurrencyUSD(selectedMarket.high_24h)}
          />
          <PriceCard
            title="24h Low"
            value={formatCurrencyUSD(selectedMarket.low_24h)}
          />
        </div>
      )}

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
                  formatter={(value: number) => [formatCurrencyUSD(value), 'Price']}
                />
                <Line
                  type="monotone"
                  dataKey="close"
                  stroke="#58a6ff"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
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
                <Bar dataKey="volume" fill="#3fb950" radius={[2, 2, 0, 0]} isAnimationActive={false} />
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
              {pricesData?.markets?.map((market) => (
                <tr
                  key={market.symbol}
                  className="border-b border-border/50 hover:bg-border/20 cursor-pointer"
                  onClick={() => setSelectedSymbol(market.symbol)}
                >
                  <td className="py-3 px-4 font-medium text-text">{market.symbol}</td>
                  <td className="py-3 px-4 text-right text-text">{formatCurrencyUSD(market.price)}</td>
                  <td className={`py-3 px-4 text-right ${market.change_pct_24h >= 0 ? 'text-success' : 'text-danger'}`}>
                    {market.change_pct_24h >= 0 ? '+' : ''}{market.change_pct_24h.toFixed(2)}%
                  </td>
                  <td className="py-3 px-4 text-right text-text-muted">{formatCurrencyUSD(market.high_24h)}</td>
                  <td className="py-3 px-4 text-right text-text-muted">{formatCurrencyUSD(market.low_24h)}</td>
                  <td className="py-3 px-4 text-right text-text-muted">{market.volume_24h.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {pricesData?.markets?.length === 0 && (
            <EmptyState
              icon={AlertTriangle}
              title="No market symbols available"
              description="No market payload returned by backend."
            />
          )}
        </div>
      </div>
    </div>
  );
}

function PriceCard({ title, value, icon: Icon, valueColor = 'text-text' }: { title: string; value: string; icon?: React.ElementType; valueColor?: string }) {
  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-text-muted text-sm">{title}</span>
        {Icon && <Icon className={`w-5 h-5 ${valueColor === 'text-success' ? 'text-success' : valueColor === 'text-danger' ? 'text-danger' : 'text-primary'}`} />}
      </div>
      <div className={`text-xl font-bold ${valueColor}`}>{value}</div>
    </div>
  );
}
