export interface Position {
  position_id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  realized_pnl: number;
  leverage: number;
  margin_used: number;
  stop_loss?: number;
  take_profit?: number;
  opened_at: string;
  updated_at: string;
}

export interface PortfolioSummary {
  total_value: number;
  cash_balance: number;
  market_value: number;
  total_pnl: number;
  unrealized_pnl: number;
  realized_pnl: number;
  daily_pnl: number;
  daily_return_pct: number;
  total_return_pct: number;
  leverage: number;
  buying_power: number;
  num_positions: number;
}

export interface PerformanceMetrics {
  total_return: number;
  total_return_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  calmar_ratio: number;
  win_rate: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
  num_trades: number;
  num_winning_trades: number;
  num_losing_trades: number;
}

export interface PriceData {
  symbol: string;
  price: number;
  change_24h: number;
  change_pct_24h: number;
  high_24h: number;
  low_24h: number;
  volume_24h: number;
  timestamp: string;
}

export interface MarketOverview {
  timestamp: string;
  markets: PriceData[];
}

export interface CandleData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Order {
  order_id: string;
  symbol: string;
  side: string;
  order_type: string;
  quantity: number;
  price?: number;
  stop_price?: number;
  status: string;
  filled_quantity: number;
  average_price?: number;
  commission: number;
  created_at: string;
  updated_at: string;
  strategy_id?: string;
  broker: string;
  error_message?: string;
  /** Profit/Loss in base currency (for trade history) */
  pnl?: number;
  /** Profit/Loss percentage (for trade history) */
  pnl_pct?: number;
}


export interface OrderCreate {
  symbol: string;
  side: string;
  order_type?: string;
  quantity: number;
  price?: number;
  stop_price?: number;
  time_in_force?: string;
  strategy_id?: string;
  broker?: string;
}

export interface Allocation {
  by_asset_class: Record<string, number>;
  by_sector: Record<string, number>;
  by_symbol: Record<string, number>;
}

export interface HistoryEntry {
  date: string;
  value: number;
  daily_return: number;
}

export interface PortfolioHistory {
  history: HistoryEntry[];
}

export interface MarketSentiment {
  fear_greed_index: number;
  sentiment_label: string;
  trading_indicator: string;
  btc_dominance: number;
  market_momentum: number;
  last_updated: string;
}


export interface NewsItem {
  id: string;
  title: string;
  source: string;
  url: string;
  summary: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  sentiment_score: number;
  symbols: string[];
  published_at: string;
  category: string;
}

export interface NewsListResponse {
  news: NewsItem[];
  total: number;
  last_updated: string;
}

export interface NewsBySymbolResponse {
  symbol: string;
  news: NewsItem[];
  total: number;
  last_updated: string;
}
