package com.aitrading.model;

import lombok.Data;
import java.util.List;
import java.util.Map;

/**
 * Trading signal from AI analysis
 */
@Data
public class TradingSignal {
    private String symbol;
    private String action; // BUY, SELL, HOLD
    private double confidence;
    private double targetPrice;
    private double stopLoss;
    private String strategy;
    private Map<String, Object> indicators;
    private List<String> reasons;
    private long timestamp;
    private double rsi;
    private double macd;
    private double ema20;
    private double ema50;
    private double bb_upper;
    private double bb_middle;
    private double bb_lower;
}
