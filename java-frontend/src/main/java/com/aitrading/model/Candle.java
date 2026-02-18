package com.aitrading.model;

import lombok.Data;

/**
 * Candlestick data for charts
 */
@Data
public class Candle {
    private long timestamp;
    private double open;
    private double high;
    private double low;
    private double close;
    private double volume;
}
