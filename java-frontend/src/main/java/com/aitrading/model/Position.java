package com.aitrading.model;

import lombok.Data;

/**
 * Trading position
 */
@Data
public class Position {
    private String symbol;
    private String side; // LONG, SHORT
    private double quantity;
    private double entryPrice;
    private double currentPrice;
    private double unrealizedPnL;
    private double pnlPercent;
    private double stopLoss;
    private double takeProfit;
    private long entryTime;
    private String status; // OPEN, CLOSED
}
