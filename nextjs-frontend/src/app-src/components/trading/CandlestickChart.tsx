import React, { useMemo, useEffect, useState, useRef } from 'react';
import { ResponsiveContainer, ComposedChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export interface OHLCVData {
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

interface CandlestickChartProps {
    data: OHLCVData[];
    symbol: string;
    height?: number;
}

const CandlestickShape = (props: any) => {
    const { x, width, y, height, payload } = props;
    if (!payload) return null;

    const open = Number(payload.open);
    const close = Number(payload.close);
    const high = Number(payload.high);
    const low = Number(payload.low);

    if (isNaN(open) || isNaN(close) || isNaN(high) || isNaN(low) || high === low) return null;

    const isBull = close > open;
    const fill = isBull ? '#22c55e' : '#ef4444';

    // Manual scale calculation based on the Bar's y (high) and height (high-low)
    // In Recharts Y-axis, y=0 is top. So high price is at y pixels, low price is at y + height pixels.
    const getYScaling = (val: number) => {
        return y + ((high - val) / (high - low)) * height;
    };

    const highY = y;
    const lowY = y + height;
    const openY = getYScaling(open);
    const closeY = getYScaling(close);

    const topBodyY = Math.min(openY, closeY);
    const bottomBodyY = Math.max(openY, closeY);
    const bodyHeight = Math.max(1, bottomBodyY - topBodyY);
    const centerX = x + width / 2;

    return (
        <g>
            <line 
                x1={centerX} 
                y1={highY} 
                x2={centerX} 
                y2={lowY} 
                stroke={fill} 
                strokeWidth={1.5} 
            />
            <rect 
                x={x} 
                y={topBodyY} 
                width={Math.max(1, width)} 
                height={bodyHeight} 
                fill={fill} 
                stroke={fill} 
                strokeWidth={1} 
                rx={1} 
            />
        </g>
    );
};

const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload as OHLCVData;
        const isBull = data.close > data.open;
        const color = isBull ? 'text-success' : 'text-danger';

        return (
            <div className="premium-glass p-4 rounded-xl shadow-2xl border border-white/10">
                <p className="text-sm font-medium text-text-muted mb-2">{new Date(data.time).toLocaleString()}</p>
                <div className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono-num text-sm">
                    <span className="text-text-muted">Open</span>
                    <span className="text-text font-medium text-right">{data.open.toFixed(2)}</span>
                    <span className="text-text-muted">High</span>
                    <span className="text-text font-medium text-right">{data.high.toFixed(2)}</span>
                    <span className="text-text-muted">Low</span>
                    <span className="text-text font-medium text-right">{data.low.toFixed(2)}</span>
                    <span className="text-text-muted">Close</span>
                    <span className={`${color} font-bold text-right drop-shadow-md`}>{data.close.toFixed(2)}</span>
                    <span className="text-text-muted mt-2">Volume</span>
                    <span className="text-text mt-2 font-medium text-right">{(data.volume / 1000).toFixed(1)}k</span>
                </div>
            </div>
        );
    }
    return null;
};

export const CandlestickChart: React.FC<CandlestickChartProps> = ({ data, symbol, height = 400 }) => {
    const [isMounted, setIsMounted] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Wait for the container to have a non-zero width
        const checkWidth = () => {
            if (containerRef.current && containerRef.current.offsetWidth > 0) {
                setIsMounted(true);
            } else {
                requestAnimationFrame(checkWidth);
            }
        };
        const rafId = requestAnimationFrame(checkWidth);
        return () => cancelAnimationFrame(rafId);
    }, []);

    // Prepariamo i dati per Recharts: il "valore" della barra è il range [low, high]
    const chartData = useMemo(() => {
        return data.map(d => ({
            ...d,
            wickRange: [d.low, d.high]
        }));
    }, [data]);

    const minLow = Math.min(...data.map(d => d.low));
    const maxHigh = Math.max(...data.map(d => d.high));
    const padding = (maxHigh - minLow) * 0.1;

    if (!data || data.length === 0) {
        return (
            <div className="w-full premium-glass-panel flex items-center justify-center" style={{ height }}>
                <span className="text-text-muted animate-pulse">Waiting for market data...</span>
            </div>
        );
    }

    return (
        <div className="premium-glass-panel relative p-4 group" ref={containerRef}>
            {/* Background ambient glow based on last close */}
            {data.length > 0 && (
                <div className={`absolute -right-20 -top-20 w-64 h-64 rounded-full blur-[100px] opacity-10 pointer-events-none transition-colors duration-1000 
          ${data[data.length - 1].close > data[data.length - 1].open ? 'bg-success' : 'bg-danger'}`}
                />
            )}

            <div className="flex justify-between items-center mb-6 pl-2">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-white/5 border border-white/10 flex items-center justify-center font-bold text-sm">
                        {symbol.charAt(0)}
                    </div>
                    <div>
                        <h2 className="text-xl font-bold tracking-tight text-text leading-none">{symbol}</h2>
                        <span className="text-xs text-text-muted font-medium uppercase tracking-wider">Perpetual Futures</span>
                    </div>
                </div>

                {/* Realtime price indicator on chart header */}
                <div className="text-right">
                    <div className={`text-2xl font-mono-num font-bold tracking-tight
             ${data[data.length - 1].close > data[data.length - 1].open ? 'text-success glow-success' : 'text-danger glow-danger'}`}>
                        ${data[data.length - 1].close.toFixed(2)}
                    </div>
                </div>
            </div>

            <div style={{ width: '100%', height: height - 80, minHeight: height - 80 }} className="relative overflow-hidden">
                {isMounted ? (
                    <ResponsiveContainer width="100%" height={height - 80} minWidth={0} debounce={50}>
                    <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                        <XAxis
                            dataKey="time"
                            stroke="hsl(var(--text-muted))"
                            fontSize={11}
                            tickMargin={10}
                            tickFormatter={(time) => {
                                const d = new Date(time);
                                return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
                            }}
                        />
                        <YAxis
                            domain={[minLow - padding, maxHigh + padding]}
                            stroke="hsl(var(--text-muted))"
                            fontSize={11}
                            orientation="right"
                            tickFormatter={(val) => val.toFixed(0)}
                            tickMargin={10}
                        />
                        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'white', opacity: 0.05 }} />

                        {/* Fake Bar that acts as OHLC renderer */}
                        <Bar
                            dataKey="wickRange"
                            shape={<CandlestickShape />}
                            isAnimationActive={false}
                        />
                    </ComposedChart>
                </ResponsiveContainer>
                ) : (
                    <div className="w-full flex items-center justify-center opacity-10" style={{ height: height - 80 }}>
                        <div className="animate-pulse bg-white/10 w-full h-full rounded-xl" />
                    </div>
                )}
            </div>
        </div>
    );
};
