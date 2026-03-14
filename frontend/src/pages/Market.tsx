import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { marketApi } from '../services/api';
import type { MarketSentiment } from '../types';
import { TrendingUp, TrendingDown, AlertTriangle, Brain } from 'lucide-react';
import NewsFeed from '../components/NewsFeed';
import { CandlestickChart } from '../components/trading/CandlestickChart';
import { OrderBook } from '../components/trading/OrderBook';



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
    timestamp: new Date(Date.now() - (50 - i) * 3600000).toISOString(),
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

  // Generiamo un Order Book mock up realistico per dimostrazione visiva
  const mockOrderBook = useMemo(() => {
    const basePrice = selectedSymbol === 'BTCUSDT' ? 43500 : selectedSymbol === 'ETHUSDT' ? 2350 : 100;

    const generateSide = (isAsk: boolean) => {
      let total = 0;
      return Array.from({ length: 15 }).map((_, i) => {
        const distance = (i + 1) * (basePrice * 0.0005);
        const price = isAsk ? basePrice + distance : basePrice - distance;
        const size = Math.random() * (isAsk ? 2 : 5) + 0.1;
        total += size;
        return { price, size, total, depthPct: Math.min(100, (total / 50) * 100) };
      });
    };
    return { asks: generateSide(true).reverse(), bids: generateSide(false), lastPrice: basePrice };
  }, [selectedSymbol]);

  const { data: prices, isLoading: pricesLoading } = useQuery({
    queryKey: ['market-prices'],
    queryFn: marketApi.getAllPrices,
    retry: 1,
    staleTime: 30000,
  });

  const { data: candles } = useQuery({
    queryKey: ['market-candles', selectedSymbol, timeframe],
    queryFn: () => marketApi.getCandles(selectedSymbol, timeframe, 100),
    retry: 1,
    staleTime: 30000,
  });

  const { data: sentiment } = useQuery({
    queryKey: ['market-sentiment'],
    queryFn: () => marketApi.getSentiment(),
    retry: 1,
    staleTime: 60000,
    refetchInterval: 60000,
  });


  // Use fallback data only when real payload is missing.
  // Avoid false "demo data" warnings on transient query errors.
  const hasPricesData = !!(prices?.markets && prices.markets.length > 0);
  const hasCandlesData = Array.isArray(candles) && candles.length > 0;
  const pricesData = hasPricesData ? prices : fallbackPrices;
  const candlesData = hasCandlesData ? candles : fallbackCandles;
  const isUsingFallback = !hasPricesData || !hasCandlesData;

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const chartData = candlesData?.map((candle: CandleData) => ({
    time: typeof candle.timestamp === 'string' ? candle.timestamp : new Date(candle.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
    volume: candle.volume,
  })) || [];

  // Get selected market data with normalized symbol comparison
  const selectedMarket = pricesData?.markets?.find((m: MarketData) =>
    m.symbol.replace('/', '') === selectedSymbol.replace('/', '')
  );

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

      {/* Sentiment Gauge */}
      {sentiment && (
        <div className="mb-6 premium-glass-panel overflow-hidden p-5">
          <div className="flex items-center gap-2 mb-4">
            <div className="p-2 rounded-lg bg-primary/20 border border-primary/30 glow-primary">
              <Brain className="w-5 h-5 text-primary" />
            </div>
            <h2 className="text-lg font-semibold text-text tracking-wide">Market Sentiment</h2>
          </div>
          <SentimentGauge sentiment={sentiment} />
        </div>
      )}

      {/* Symbol Selector */}
      <div className="flex gap-4 mb-6">
        <div className="flex gap-2">

          {symbols.map((symbol) => (
            <button
              key={symbol}
              onClick={() => setSelectedSymbol(symbol)}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-300 ${selectedSymbol === symbol
                ? 'bg-primary text-white glow-primary border-primary shadow-lg shadow-primary/20'
                : 'bg-white/[0.03] border border-white/[0.08] text-text-muted hover:text-text hover:bg-white/[0.08]'
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
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${timeframe === tf
                ? 'bg-primary/20 text-primary border-primary/50'
                : 'bg-white/[0.02] border border-transparent text-text-muted hover:text-text hover:bg-white/[0.05]'
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
            value={formatCurrency(selectedMarket.price)}
            icon={selectedMarket.change_pct_24h >= 0 ? TrendingUp : TrendingDown}
          />
          <PriceCard
            title="24h Change"
            value={`${selectedMarket.change_pct_24h >= 0 ? '+' : ''}${selectedMarket.change_pct_24h.toFixed(2)}%`}
            valueColor={selectedMarket.change_pct_24h >= 0 ? 'text-success' : 'text-danger'}
          />
          <PriceCard
            title="24h High"
            value={formatCurrency(selectedMarket.high_24h)}
          />
          <PriceCard
            title="24h Low"
            value={formatCurrency(selectedMarket.low_24h)}
          />
        </div>
      )}

      {/* Advanced Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">

        {/* Candlestick Chart (Span 3 cols) */}
        <div className="lg:col-span-3">
          <CandlestickChart
            data={chartData}
            symbol={selectedSymbol}
            height={480}
          />
        </div>

        {/* Order Book (Span 1 col) */}
        <div className="lg:col-span-1 h-[480px]">
          <OrderBook
            bids={mockOrderBook.bids}
            asks={mockOrderBook.asks}
            lastPrice={mockOrderBook.lastPrice}
            symbol={selectedSymbol}
          />
        </div>
      </div>

      {/* News Feed */}
      <div className="mt-6">
        <NewsFeed limit={6} />
      </div>

      {/* All Markets Table */}
      <div className="mt-6 premium-glass-panel overflow-hidden">
        <div className="px-6 py-5 border-b border-white/[0.05] bg-white/[0.02]">
          <h2 className="text-lg font-semibold text-text tracking-wide flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-primary glow-primary"></span>
            All Markets
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-black/20">
              <tr>
                <th className="text-left py-4 px-6 text-xs text-text-muted font-semibold uppercase tracking-wider">Symbol</th>
                <th className="text-right py-4 px-6 text-xs text-text-muted font-semibold uppercase tracking-wider">Price</th>
                <th className="text-right py-4 px-6 text-xs text-text-muted font-semibold uppercase tracking-wider">24h Change</th>
                <th className="text-right py-4 px-6 text-xs text-text-muted font-semibold uppercase tracking-wider">24h High</th>
                <th className="text-right py-4 px-6 text-xs text-text-muted font-semibold uppercase tracking-wider">24h Low</th>
                <th className="text-right py-4 px-6 text-xs text-text-muted font-semibold uppercase tracking-wider">Volume</th>
              </tr>
            </thead>
            <tbody>
              {pricesData?.markets?.map((market: MarketData) => (
                <tr
                  key={market.symbol}
                  className={`border-b border-white/[0.05] hover:bg-white/[0.04] cursor-pointer transition-colors ${selectedSymbol === market.symbol ? 'bg-primary/5' : ''}`}
                  onClick={() => setSelectedSymbol(market.symbol)}
                >
                  <td className="py-4 px-6 font-semibold text-text flex items-center gap-2">
                    <div className="w-6 h-6 rounded-full bg-white/10 flex items-center justify-center text-xs">
                      {market.symbol.charAt(0)}
                    </div>
                    {market.symbol.replace('USDT', '')}
                  </td>
                  <td className="py-4 px-6 text-right text-text font-mono-num text-lg">{formatCurrency(market.price)}</td>
                  <td className={`py-4 px-6 text-right font-mono-num ${market.change_pct_24h >= 0 ? 'text-success drop-shadow-[0_0_5px_rgba(34,197,94,0.3)]' : 'text-danger drop-shadow-[0_0_5px_rgba(239,68,68,0.3)]'}`}>
                    <div className="bg-black/20 inline-block px-2 py-1 rounded-md">
                      {market.change_pct_24h >= 0 ? '+' : ''}{market.change_pct_24h.toFixed(2)}%
                    </div>
                  </td>
                  <td className="py-4 px-6 text-right text-text-muted font-mono-num">{formatCurrency(market.high_24h)}</td>
                  <td className="py-4 px-6 text-right text-text-muted font-mono-num">{formatCurrency(market.low_24h)}</td>
                  <td className="py-4 px-6 text-right text-text-muted font-mono-num">{(market.volume_24h / 1000000).toFixed(2)}M</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function PriceCard({ title, value, icon: Icon, valueColor = 'text-text' }: { title: string; value: string; icon?: React.ElementType; valueColor?: string }) {
  return (
    <div className="premium-glass-panel p-5 premium-glass-hover">
      <div className="flex items-center justify-between mb-3">
        <span className="text-text-muted text-sm font-medium uppercase tracking-wider">{title}</span>
        {Icon && <Icon className={`w-5 h-5 ${valueColor === 'text-success' ? 'text-success' : valueColor === 'text-danger' ? 'text-danger' : 'text-primary'}`} />}
      </div>
      <div className={`text-2xl font-bold font-mono-num ${valueColor}`}>{value}</div>
    </div>
  );
}

function SentimentGauge({ sentiment }: { sentiment: MarketSentiment }) {
  const {
    fear_greed_index = 50,
    sentiment_label = 'Neutral',
    sentiment_emoji = '😐',
    btc_dominance = 0,
    market_momentum = 0
  } = sentiment || {};

  // Handle undefined values safely
  const fgi = typeof fear_greed_index === 'number' ? fear_greed_index : 50;
  const btc = typeof btc_dominance === 'number' ? btc_dominance : 0;
  const momentum = typeof market_momentum === 'number' ? market_momentum : 0;

  // Calculate color based on index (0-100)
  const getColor = (index: number) => {
    if (index <= 20) return 'text-red-500';      // Extreme Fear
    if (index <= 40) return 'text-orange-500';   // Fear
    if (index <= 60) return 'text-yellow-500';   // Neutral
    if (index <= 80) return 'text-green-500';    // Greed
    return 'text-emerald-500';                   // Extreme Greed
  };

  const getBgColor = (index: number) => {
    if (index <= 20) return 'bg-red-500';
    if (index <= 40) return 'bg-orange-500';
    if (index <= 60) return 'bg-yellow-500';
    if (index <= 80) return 'bg-green-500';
    return 'bg-emerald-500';
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Fear & Greed Index Gauge */}
      <div className="bg-background border border-border rounded-lg p-4">
        <div className="text-center">
          <div className="text-3xl mb-2">{sentiment_emoji}</div>
          <div className={`text-4xl font-bold ${getColor(fgi)}`}>
            {fgi}
          </div>
          <div className="text-text-muted text-sm mt-1">Fear & Greed Index</div>
          <div className={`font-medium mt-2 ${getColor(fgi)}`}>
            {sentiment_label}
          </div>
          {/* Progress bar */}
          <div className="mt-3 h-2 bg-border rounded-full overflow-hidden">
            <div
              className={`h-full ${getBgColor(fgi)} transition-all duration-500`}
              style={{ width: `${fgi}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-text-muted mt-1">
            <span>Extreme Fear</span>
            <span>Extreme Greed</span>
          </div>
        </div>
      </div>

      {/* BTC Dominance */}
      <div className="bg-background border border-border rounded-lg p-4">
        <div className="text-center">
          <div className="text-3xl mb-2">₿</div>
          <div className="text-4xl font-bold text-primary">
            {btc.toFixed(1)}%
          </div>
          <div className="text-text-muted text-sm mt-1">BTC Dominance</div>
          <div className="text-text text-sm mt-2">
            Market share of Bitcoin
          </div>
        </div>
      </div>

      {/* Market Momentum */}
      <div className="bg-background border border-border rounded-lg p-4">
        <div className="text-center">
          <div className="text-3xl mb-2">📈</div>
          <div className={`text-4xl font-bold ${momentum >= 0 ? 'text-success' : 'text-danger'}`}>
            {momentum >= 0 ? '+' : ''}{momentum.toFixed(2)}%
          </div>
          <div className="text-text-muted text-sm mt-1">Market Momentum</div>
          <div className="text-text text-sm mt-2">
            {momentum >= 0 ? 'Bullish trend' : 'Bearish trend'}
          </div>
        </div>
      </div>

      {/* Sentiment Summary */}
      <div className="bg-background border border-border rounded-lg p-4">
        <div className="text-center">
          <div className="text-3xl mb-2">🎯</div>
          <div className="text-lg font-bold text-text">
            {fgi <= 40 ? 'Consider Buying' : fgi >= 60 ? 'Consider Selling' : 'Hold Position'}
          </div>
          <div className="text-text-muted text-sm mt-1">AI Recommendation</div>
          <div className="text-text text-xs mt-2">
            Based on Fear & Greed Index
          </div>
        </div>
      </div>
    </div>
  );
}
