package com.aitrading.model;

import lombok.Data;
import java.util.List;
import java.util.Map;

/**
 * Portfolio information
 */
@Data
public class Portfolio {
    private double totalValue;
    private double cashBalance;
    private double unrealizedPnL;
    private double realizedPnL;
    private double winRate;
    private double SharpeRatio;
    private double maxDrawdown;
    private List<Position> positions;
    private List<Trade> recentTrades;
    private Map<String, Double> allocation;
    private long timestamp;
}
