import axios from 'axios';
import type {
  PortfolioSummary,
  Position,
  PerformanceMetrics,
  Allocation,
  PortfolioHistory,
  MarketOverview,
  PriceData,
  CandleData,
  Order,
  OrderCreate,
  EmergencyStatus,
  MarketSentiment,
} from '../types';

// Use environment variable for API base URL
// In production (Vercel), requests to /api/v1/* are proxied to Render backend
// In local development, use localhost:8000
const defaultApiBase =
  typeof window !== 'undefined' && ['5173', '3000'].includes(window.location.port)
    ? 'http://localhost:8000/api/v1'
    : '/api/v1';

const API_BASE = import.meta.env.VITE_API_BASE_URL || defaultApiBase;

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  // Add timeout and retry logic
  timeout: 10000,
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === 'ERR_NETWORK') {
      console.error('Network error: Cannot connect to backend. Is your local backend running?');
    }
    return Promise.reject(error);
  }
);

// Portfolio API
export const portfolioApi = {
  getSummary: async (): Promise<PortfolioSummary> => {
    const { data } = await api.get<PortfolioSummary>('/portfolio/summary');
    return data;
  },

  getDualSummary: async () => {
    const { data } = await api.get('/portfolio/summary/dual');
    return data;
  },

  getPositions: async (symbol?: string): Promise<Position[]> => {
    const params = symbol ? { symbol } : {};
    const { data } = await api.get<Position[]>('/portfolio/positions', { params });
    return data;
  },

  getPerformance: async (): Promise<PerformanceMetrics> => {
    const { data } = await api.get<PerformanceMetrics>('/portfolio/performance');
    return data;
  },

  getAllocation: async (): Promise<Allocation> => {
    const { data } = await api.get<Allocation>('/portfolio/allocation');
    return data;
  },

  getHistory: async (days: number = 30): Promise<PortfolioHistory> => {
    const { data } = await api.get<PortfolioHistory>('/portfolio/history', {
      params: { days },
    });
    return data;
  },
};

// Market API
export const marketApi = {
  getAllPrices: async (): Promise<MarketOverview> => {
    const { data } = await api.get<MarketOverview>('/market/prices');
    return data;
  },

  getPrice: async (symbol: string): Promise<PriceData> => {
    const { data } = await api.get<PriceData>(`/market/price/${symbol}`);
    return data;
  },

  getCandles: async (
    symbol: string,
    interval: string = '1h',
    limit: number = 100
  ): Promise<CandleData[]> => {
    // Convert symbol format from "BTC/USDT" to "BTCUSDT" if needed
    const normalizedSymbol = symbol.replace('/', '');
    const { data } = await api.get<CandleData[]>(`/market/candles/${normalizedSymbol}`, {
      params: { interval, limit },
    });
    return data;
  },

  getOrderBook: async (symbol: string) => {
    const { data } = await api.get(`/market/orderbook/${symbol}`);
    return data;
  },

  getSentiment: async (): Promise<MarketSentiment> => {
    const { data } = await api.get<MarketSentiment>('/market/sentiment');
    return data;
  },
};

// Orders API
export const ordersApi = {
  list: async (symbol?: string, status?: string): Promise<Order[]> => {
    const params: Record<string, string> = {};
    if (symbol) params.symbol = symbol;
    if (status) params.status = status;
    const { data } = await api.get<Order[]>('/orders', { params });
    return data;
  },

  get: async (orderId: string): Promise<Order> => {
    const { data } = await api.get<Order>(`/orders/${orderId}`);
    return data;
  },

  create: async (order: OrderCreate): Promise<Order> => {
    const { data } = await api.post<Order>('/orders', order);
    return data;
  },

  cancel: async (orderId: string): Promise<void> => {
    await api.delete(`/orders/${orderId}`);
  },

  execute: async (orderId: string): Promise<Order> => {
    const { data } = await api.post<Order>(`/orders/${orderId}/execute`);
    return data;
  },
};

export const emergencyApi = {
  getStatus: async (): Promise<EmergencyStatus> => {
    const { data } = await api.get<EmergencyStatus>('/orders/emergency/status');
    return data;
  },

  activate: async (reason: string, adminKey: string): Promise<EmergencyStatus> => {
    const { data } = await api.post<EmergencyStatus>('/orders/emergency/activate', {
      confirm: true,
      reason,
    }, {
      headers: {
        'X-Admin-Key': adminKey,
        'X-Admin-User': 'ui-operator',
      },
    });
    return data;
  },

  deactivate: async (reason: string, adminKey: string): Promise<EmergencyStatus> => {
    const { data } = await api.post<EmergencyStatus>('/orders/emergency/deactivate', {
      confirm: true,
      reason,
    }, {
      headers: {
        'X-Admin-Key': adminKey,
        'X-Admin-User': 'ui-operator',
      },
    });
    return data;
  },
};

// Risk API
export interface RiskMetrics {
  var_1d: number;
  var_5d: number;
  cvar_1d: number;
  cvar_5d: number;
  volatility: number;
  beta: number;
  correlation_to_btc: number;
  max_drawdown: number;
  sharpe_ratio: number;
  leverage: number;
  margin_utilization: number;
}

export interface RiskLimit {
  limit_id: string;
  limit_type: string;
  limit_value: number;
  current_value: number;
  limit_percentage: number;
  is_breached: boolean;
  severity: string;
}

export interface PositionRisk {
  symbol: string;
  position_size: number;
  market_value: number;
  var_contribution: number;
  beta_weighted_exposure: number;
  correlation_to_portfolio: number;
  concentration_risk: number;
}

export const riskApi = {
  getMetrics: async (): Promise<RiskMetrics> => {
    const { data } = await api.get<RiskMetrics>('/risk/metrics');
    return data;
  },
  getLimits: async (): Promise<RiskLimit[]> => {
    const { data } = await api.get<RiskLimit[]>('/risk/limits');
    return data;
  },
  getPositionRisks: async (): Promise<PositionRisk[]> => {
    const { data } = await api.get<PositionRisk[]>('/risk/positions');
    return data;
  },
  getCorrelationMatrix: async () => {
    const { data } = await api.get('/risk/correlation');
    return data;
  },
};

// Strategy API
export interface Strategy {
  strategy_id: string;
  name: string;
  description: string;
  strategy_type: string;
  asset_classes: string[];
  parameters: Record<string, any>;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface StrategyPerformance {
  strategy_id: string;
  strategy_name: string;
  total_return: number;
  total_return_pct: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  num_signals: number;
  num_trades: number;
  avg_trade_pnl: number;
}

export const strategyApi = {
  list: async (params?: { strategy_type?: string; enabled_only?: boolean }): Promise<Strategy[]> => {
    const { data } = await api.get<Strategy[]>('/strategy/', { params });
    return data;
  },
  get: async (id: string): Promise<Strategy> => {
    const { data } = await api.get<Strategy>(`/strategy/${id}`);
    return data;
  },
  getPerformance: async (id: string): Promise<StrategyPerformance> => {
    const { data } = await api.get<StrategyPerformance>(`/strategy/${id}/performance`);
    return data;
  },
  run: async (id: string) => {
    const { data } = await api.post(`/strategy/${id}/run`);
    return data;
  }
};

// News API
export const newsApi = {
  getNews: async (params?: { limit?: number; refresh?: string }) => {
    const { data } = await api.get('/news', { params });
    return data;
  },
  getNewsBySymbol: async (symbol: string, limit?: number, refresh?: string) => {
    const { data } = await api.get(`/news/${symbol}`, { params: { limit, refresh } });
    return data;
  },
};

// Payment API
export const paymentApi = {
  isConfigured: () => {
    return !!import.meta.env.VITE_STRIPE_PUBLIC_KEY;
  },
  redirectToPaymentLink: () => {
    const paymentLink = import.meta.env.VITE_STRIPE_PAYMENT_LINK;
    if (paymentLink) {
      window.location.href = paymentLink;
    }
  },
  createCheckoutSession: async (email: string) => {
    const { data } = await api.post('/payments/stripe/checkout-session', { email });
    return data;
  },
};

// Cache API
export interface CacheStats {
  in_memory: {
    size: number;
    hits: number;
    misses: number;
    hit_rate: number;
    available: boolean;
  };
  redis: {
    connected: boolean;
    keys: number;
    memory: string;
  };
}

export interface CacheClearResponse {
  success: boolean;
  message: string;
  cleared_count: number;
}

export const cacheApi = {
  /**
   * Get cache statistics from all cache backends
   */
  getStats: async (): Promise<CacheStats> => {
    const { data } = await api.get<CacheStats>('/cache/');
    return data;
  },

  /**
   * Clear all caches (in-memory and Redis)
   */
  clearAll: async (): Promise<CacheClearResponse> => {
    const { data } = await api.delete<CacheClearResponse>('/cache/');
    return data;
  },

  /**
   * Clear in-memory cache only
   */
  clearInMemory: async (): Promise<CacheClearResponse> => {
    const { data } = await api.delete<CacheClearResponse>('/cache/in-memory');
    return data;
  },

  /**
   * Clear Redis cache only
   */
  clearRedis: async (pattern: string = ''): Promise<CacheClearResponse> => {
    const { data } = await api.delete<CacheClearResponse>('/cache/redis', {
      params: { pattern },
    });
    return data;
  },
};

// Health API
export interface HealthStatus {
  status: string;
  timestamp: string;
  service: string;
  version: string;
  environment: string;
}

export const healthApi = {
  getStatus: async (): Promise<HealthStatus> => {
    const { data } = await api.get<HealthStatus>('/health');
    return data;
  },
};

export { api };
export default api;
