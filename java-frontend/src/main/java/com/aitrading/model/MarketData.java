package com.aitrading.model;

import lombok.Data;
import java.util.List;
import java.util.Map;

/**
 * Market data response from Python backend
 */
@Data
public class MarketData {
    private String symbol;
    private double price;
    private double change24h;
    private double changePercent24h;
    private double high24h;
    private double low24h;
    private double volume24h;
    private long timestamp;
    private List<Candle> candles;
    private Map<String, Object> indicators;
}
