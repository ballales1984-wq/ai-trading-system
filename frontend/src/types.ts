export interface PortfolioSummary {
  totalValue: number;
  cash: number;
  positionsValue: number;
  pnl: number;
  pnlPct: number;
  dailyPnL: number;
  dailyPnLPct: number;
}

export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_pct: number;
  unrealized_pnl: number;
  side: 'BUY' | 'SELL';
  position_id: string;
  market_value: number;
}

export interface PerformanceMetrics {
  total_return_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown_pct: number;
  win_rate: number;
  total_trades: number;
  profit_factor: number;
  num_winning_trades: number;
  num_losing_trades: number;
}

export interface Allocation {
  by_symbol: {
    symbol: string;
    allocationPct: number;
    value: number;
  }[];
}

export interface PortfolioHistory {
  history: {
    date: string;
    value: number;
  }[];
}

export interface MarketOverview {
  markets: Array<{
    symbol: string;
    price: number;
    change_pct_24h: number;
    high_24h: number;
    low_24h: number;
    volume_24h: number;
  }>;
}

export interface PriceData {
  symbol: string;
  price: number;
  timestamp: string;
}

export interface CandleData {
  timestamp: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Order {
  order_id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  order_type: 'market' | 'limit';
  quantity: number;
  average_price: number;
  status: 'pending' | 'filled' | 'cancelled';
  created_at: string;
}

export interface OrderCreate {
  symbol: string;
  side: 'BUY' | 'SELL';
  order_type: 'market' | 'limit';
  quantity: number;
  price?: number;
}

export interface EmergencyStatus {
  active: boolean;
  reason: string;
  activated_at: string;
  activated_by: string;
  trading_halted: boolean;
}

export interface MarketSentiment {
  fear_greed_index?: number;
  sentiment_label?: string;
  sentiment_emoji?: string;
  btc_dominance?: number;
  market_momentum?: number;
}

export interface CorrelationMatrix {
  assets: string[];
  matrix: number[][];
}

export interface NewsItem {
  id: number;
  title: string;
  content: string;
  source: string;
  sentiment?: 'positive' | 'negative' | 'neutral';
  published_at: string;
  symbols?: string[];
  summary?: string;
  category?: string;
  url?: string;
}
