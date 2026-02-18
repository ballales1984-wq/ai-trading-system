package com.aitrading.model;

import lombok.Data;

/**
 * Completed trade record
 */
@Data
public class Trade {
    private String id;
    private String symbol;
    private String side; // BUY, SELL
    private double quantity;
    private double price;
    private double pnl;
    private double pnlPercent;
    private String strategy;
    private long entryTime;
    private long exitTime;
    private String status; // OPEN, CLOSED
}
