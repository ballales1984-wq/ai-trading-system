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
  MarketSentiment,
  NewsListResponse,
  NewsBySymbolResponse,
} from '../types';



// Use environment variable for API base URL
// In production (Vercel), set VITE_API_BASE_URL to your backend URL
// Local: http://localhost:8000/api/v1 (via vite proxy)
// Vercel: The API is served from the same domain, so use relative path
const isLocalhost = typeof window !== 'undefined' && ['5173', '3000'].includes(window.location.port);

let API_BASE: string;
if (isLocalhost) {
  // Local development - use localhost:8000
  API_BASE = 'http://localhost:8000/api/v1';
} else {
  // Production (Vercel) - use relative path to call serverless API
  API_BASE = '/api/v1';
}


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
    const { data } = await api.get<CandleData[]>(`/market/candles/${symbol}`, {
      params: { interval, limit },
    });
    return data;
  },

  getOrderBook: async (symbol: string) => {
    const { data } = await api.get(`/market/orderbook/${symbol}`);
    return data;
  },

  /**
   * Get market sentiment data (Fear & Greed Index)
   * Returns fear/greed index, sentiment label with emoji, BTC dominance, and market momentum
   */
  getSentiment: async (): Promise<MarketSentiment> => {
    const { data } = await api.get<MarketSentiment>('/market/sentiment');
    return data;
  },
};


// Risk API
export const riskApi = {
  getMetrics: async (): Promise<{
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
  }> => {
    const { data } = await api.get('/risk/metrics');
    return data;
  },

  getCorrelationMatrix: async (): Promise<{
    assets: string[];
    matrix: number[][];
  }> => {
    const { data } = await api.get('/risk/correlation');
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

  /**
   * Get trade history with P&L data
   * Supports filtering by symbol, status, and date range
   */
  getHistory: async (params?: {
    symbol?: string;
    status?: string;
    dateFrom?: string;
    dateTo?: string;
    limit?: number;
  }): Promise<Order[]> => {
    const { data } = await api.get<Order[]>('/orders/history', { params });
    return data;
  },
};


// Stripe Payment API
export interface CreateCheckoutRequest {
  email?: string;
  price_id?: string;
  quantity?: number;
}

export interface CreateCheckoutResponse {
  checkout_url: string;
  session_id: string;
}

export const paymentApi = {
  /**
   * Create a Stripe checkout session
   * Uses backend API to generate Stripe checkout URL
   */
  createCheckoutSession: async (payload: CreateCheckoutRequest): Promise<CreateCheckoutResponse> => {
    const { data } = await api.post<CreateCheckoutResponse>(
      '/payments/stripe/checkout-session',
      payload
    );
    return data;
  },

  /**
   * Redirect to Stripe checkout
   * Creates a checkout session and redirects the user
   */
  redirectToCheckout: async (email?: string, priceId?: string): Promise<void> => {
    const response = await paymentApi.createCheckoutSession({
      email,
      price_id: priceId,
      quantity: 1,
    });
    
    // Redirect to Stripe checkout
    window.location.href = response.checkout_url;
  },

  /**
   * Get Stripe payment link from environment
   * For simple payment link redirect (no backend required)
   */
  getPaymentLink: (): string | undefined => {
    return import.meta.env.VITE_STRIPE_PAYMENT_LINK;
  },

  /**
   * Check if Stripe payment is configured
   */
  isConfigured: (): boolean => {
    return !!paymentApi.getPaymentLink();
  },

  /**
   * Redirect to Stripe payment link
   */
  redirectToPaymentLink: (): void => {
    const paymentLink = paymentApi.getPaymentLink();
    if (paymentLink) {
      window.location.href = paymentLink;
    } else {
      console.error('Stripe payment link not configured');
    }
  },
};

// News API
export const newsApi = {
  /**
   * Get latest crypto news feed
   * Returns news items with sentiment analysis and related symbols
   */
  getNews: async (params?: {
    limit?: number;
    sentiment?: string;
    category?: string;
  }): Promise<NewsListResponse> => {
    const { data } = await api.get<NewsListResponse>('/news', { params });
    return data;
  },

  /**
   * Get news filtered by trading symbol
   * Returns news items related to a specific cryptocurrency
   */
  getNewsBySymbol: async (
    symbol: string,
    limit?: number
  ): Promise<NewsBySymbolResponse> => {
    const { data } = await api.get<NewsBySymbolResponse>(`/news/${symbol}`, {
      params: { limit },
    });
    return data;
  },
};

export default api;
