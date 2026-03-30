import React from 'react';

export interface OrderBookLevel {
    price: number;
    size: number;
    total: number;
    depthPct: number; // 0 to 100 for background depth bar
}

interface OrderBookProps {
    bids: OrderBookLevel[];
    asks: OrderBookLevel[];
    lastPrice?: number;
    symbol: string;
}

export const OrderBook: React.FC<OrderBookProps> = ({ bids, asks, lastPrice, symbol }) => {
    return (
        <div className="premium-glass-panel overflow-hidden flex flex-col h-full font-mono-num text-sm">
            <div className="px-4 py-3 border-b border-white/[0.05] flex justify-between items-center bg-black/20">
                <h3 className="font-semibold text-text tracking-wide text-sm flex items-center gap-2">
                    Order Book
                    <span className="text-xs text-text-muted px-2 py-0.5 bg-white/5 rounded-md border border-white/10">
                        {symbol.replace('USDT', '')}
                    </span>
                </h3>
                <span className="text-xs text-text-muted">Size Base</span>
            </div>

            <div className="flex px-4 py-2 text-xs font-semibold text-text-muted uppercase tracking-wider bg-white/[0.02]">
                <div className="flex-1 text-left">Price (USD)</div>
                <div className="flex-1 text-right">Size</div>
                <div className="flex-1 text-right">Total</div>
            </div>

            <div className="flex-1 overflow-y-auto space-y-px p-1">
                {/* ASKS (Sells) - Order reversed to show lowest ask at bottom */}
                <div className="flex flex-col-reverse">
                    {asks.slice(0, 15).map((ask, i) => (
                        <div key={`ask-${i}`} className="flex relative justify-between px-3 py-1 hover:bg-white/[0.04] group cursor-default">
                            {/* Depth Background */}
                            <div
                                className="absolute right-0 top-0 bottom-0 bg-danger/10 transition-all duration-300"
                                style={{ width: `${ask.depthPct}%` }}
                            />
                            <div className="flex-1 text-danger font-medium z-10 transition-colors group-hover:text-red-400">
                                {ask.price.toFixed(2)}
                            </div>
                            <div className="flex-1 text-right text-text z-10">{ask.size.toFixed(4)}</div>
                            <div className="flex-1 text-right text-text-muted z-10 opacity-70">{ask.total.toFixed(4)}</div>
                        </div>
                    ))}
                </div>

                {/* Spread / Last Price Indicator */}
                <div className="flex items-center justify-between px-4 py-3 bg-black/30 border-y border-white/[0.05] my-2">
                    {lastPrice ? (
                        <div className="text-xl font-bold flex items-center gap-2">
                            <span className="text-text">${lastPrice.toFixed(2)}</span>
                            <span className="text-xs px-1.5 py-0.5 rounded bg-primary/20 text-primary border border-primary/30">LTP</span>
                        </div>
                    ) : (
                        <span className="text-text-muted animate-pulse">Waiting for price...</span>
                    )}

                    {asks.length > 0 && bids.length > 0 && (
                        <div className="text-xs text-text-muted flex flex-col items-end">
                            <span>Spread</span>
                            <span className="font-medium text-warning glow-warning">
                                ${(asks[0].price - bids[0].price).toFixed(2)}
                            </span>
                        </div>
                    )}
                </div>

                {/* BIDS (Buys) */}
                <div>
                    {bids.slice(0, 15).map((bid, i) => (
                        <div key={`bid-${i}`} className="flex relative justify-between px-3 py-1 hover:bg-white/[0.04] group cursor-default">
                            {/* Depth Background */}
                            <div
                                className="absolute right-0 top-0 bottom-0 bg-success/10 transition-all duration-300"
                                style={{ width: `${bid.depthPct}%` }}
                            />
                            <div className="flex-1 text-success font-medium z-10 transition-colors group-hover:text-green-400">
                                {bid.price.toFixed(2)}
                            </div>
                            <div className="flex-1 text-right text-text z-10">{bid.size.toFixed(4)}</div>
                            <div className="flex-1 text-right text-text-muted z-10 opacity-70">{bid.total.toFixed(4)}</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
