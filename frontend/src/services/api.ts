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
} from '../types';

const API_BASE = '/api/v1';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

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

export default api;

