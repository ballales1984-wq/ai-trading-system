/**
 * useMarketData — Real-time market prices via WebSocket
 * Falls back to the REST polling from React Query if WS is unavailable.
 *
 * Messages expected from the backend WS /ws/prices:
 *   { type: "price_update", data: { symbol: string, price: number, change_pct_24h: number, ... } }
 *   { type: "portfolio_update", data: { total_value: number, daily_pnl: number, ... } }
 */
import { useState, useCallback } from 'react';
import { useWebSocket, type WsStatus } from './useWebSocket';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface LivePriceData {
    symbol: string;
    price: number;
    change_pct_24h: number;
    volume_24h?: number;
    updated_at: string;
}

export interface LivePortfolioData {
    total_value: number;
    daily_pnl: number;
    unrealized_pnl: number;
    num_positions: number;
    daily_return_pct: number;
}

interface WsMessage {
    type: 'price_update' | 'portfolio_update' | 'ping' | string;
    data: unknown;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

/**
 * Returns live prices (keyed by symbol) and live portfolio data,
 * plus the WebSocket connection status.
 */
export function useMarketData() {
    const [prices, setPrices] = useState<Record<string, LivePriceData>>({});
    const [portfolio, setPortfolio] = useState<LivePortfolioData | null>(null);
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

    // Derive WS url - use relative path to leverage Vite proxy
    const wsUrl = (() => {
        if (typeof window === 'undefined') return '';
        const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
        return `${proto}://${window.location.host}/ws/prices`;
    })();

    const handleMessage = useCallback((raw: unknown) => {
        const msg = raw as WsMessage;

        if (msg.type === 'price_update') {
            const item = msg.data as LivePriceData;
            setPrices((prev) => ({
                ...prev,
                [item.symbol]: item,
            }));
            setLastUpdated(new Date());
        } else if (msg.type === 'portfolio_update') {
            setPortfolio(msg.data as LivePortfolioData);
            setLastUpdated(new Date());
        }
        // 'ping' frames are ignored
    }, []);

    const { status, send } = useWebSocket({
        url: wsUrl,
        onMessage: handleMessage,
        enabled: !!wsUrl,
        reconnectMaxAttempts: 6,
    });

    return {
        /** Live prices keyed by symbol, e.g. prices["BTCUSDT"].price */
        prices,
        /** Live aggregated portfolio snapshot */
        portfolio,
        /** ISO timestamp of last WS message received */
        lastUpdated,
        /** WS connection status */
        wsStatus: status as WsStatus,
        /** Whether the WS is connected and streaming */
        isLive: status === 'open',
        /** Send a raw message to the backend WS */
        send,
    };
}
