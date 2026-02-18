package com.aitrading.service;

import com.aitrading.model.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import java.util.*;

import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Service to communicate with Python backend APIs
 */
@Service
public class TradingApiService {
    
    @Value("${python.api.baseurl:http://localhost:5000}")
    private String pythonApiBaseUrl;
    
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    
    public TradingApiService() {
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }
    
    /**
     * Get market data for a symbol
     */
    public MarketData getMarketData(String symbol) {
        try {
            String url = pythonApiBaseUrl + "/api/market/" + symbol;
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return objectMapper.convertValue(response.getBody(), MarketData.class);
            }
        } catch (Exception e) {
            System.err.println("Error fetching market data: " + e.getMessage());
        }
        return createMockMarketData(symbol);
    }
    
    /**
     * Get all available symbols
     */
    public List<String> getSymbols() {
        try {
            String url = pythonApiBaseUrl + "/api/symbols";
            ResponseEntity<List> response = restTemplate.getForEntity(url, List.class);
            if (response.getStatusCode() == HttpStatus.OK) {
                return response.getBody();
            }
        } catch (Exception e) {
            System.err.println("Error fetching symbols: " + e.getMessage());
        }
        return Arrays.asList("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT");
    }
    
    /**
     * Get trading signals
     */
    public List<TradingSignal> getSignals() {
        try {
            String url = pythonApiBaseUrl + "/api/signals";
            ResponseEntity<List> response = restTemplate.getForEntity(url, List.class);
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return objectMapper.convertValue(response.getBody(), 
                    objectMapper.getTypeFactory().constructCollectionType(List.class, TradingSignal.class));
            }
        } catch (Exception e) {
            System.err.println("Error fetching signals: " + e.getMessage());
        }
        return createMockSignals();
    }
    
    /**
     * Get portfolio data
     */
    public Portfolio getPortfolio() {
        try {
            String url = pythonApiBaseUrl + "/api/portfolio";
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return objectMapper.convertValue(response.getBody(), Portfolio.class);
            }
        } catch (Exception e) {
            System.err.println("Error fetching portfolio: " + e.getMessage());
        }
        return createMockPortfolio();
    }
    
    /**
     * Execute a trade
     */
    public Map<String, Object> executeTrade(String symbol, String side, double quantity) {
        try {
            String url = pythonApiBaseUrl + "/api/trade";
            Map<String, Object> request = new HashMap<>();
            request.put("symbol", symbol);
            request.put("side", side);
            request.put("quantity", quantity);
            
            ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);
            return response.getBody();
        } catch (Exception e) {
            System.err.println("Error executing trade: " + e.getMessage());
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("message", e.getMessage());
            return error;
        }
    }
    
    // Mock data methods for demonstration
    private MarketData createMockMarketData(String symbol) {
        MarketData data = new MarketData();
        data.setSymbol(symbol);
        data.setPrice(getMockPrice(symbol));
        data.setChange24h(getMockPrice(symbol) * 0.02);
        data.setChangePercent24h(2.0);
        data.setHigh24h(getMockPrice(symbol) * 1.05);
        data.setLow24h(getMockPrice(symbol) * 0.95);
        data.setVolume24h(1000000.0);
        data.setTimestamp(System.currentTimeMillis());
        return data;
    }
    
    private double getMockPrice(String symbol) {
        switch (symbol) {
            case "BTCUSDT": return 52000.0;
            case "ETHUSDT": return 2800.0;
            case "BNBUSDT": return 320.0;
            case "SOLUSDT": return 120.0;
            case "XRPUSDT": return 0.55;
            default: return 100.0;
        }
    }
    
    private List<TradingSignal> createMockSignals() {
        List<TradingSignal> signals = new ArrayList<>();
        
        TradingSignal btc = new TradingSignal();
        btc.setSymbol("BTCUSDT");
        btc.setAction("BUY");
        btc.setConfidence(0.78);
        btc.setTargetPrice(55000.0);
        btc.setStopLoss(49000.0);
        btc.setRsi(45.2);
        btc.setMacd(125.5);
        btc.setEma20(51500.0);
        btc.setEma50(50500.0);
        btc.setTimestamp(System.currentTimeMillis());
        signals.add(btc);
        
        TradingSignal eth = new TradingSignal();
        eth.setSymbol("ETHUSDT");
        eth.setAction("HOLD");
        eth.setConfidence(0.65);
        eth.setTargetPrice(2900.0);
        eth.setStopLoss(2600.0);
        eth.setRsi(52.1);
        eth.setMacd(15.2);
        eth.setEma20(2780.0);
        eth.setEma50(2750.0);
        eth.setTimestamp(System.currentTimeMillis());
        signals.add(eth);
        
        return signals;
    }
    
    private Portfolio createMockPortfolio() {
        Portfolio portfolio = new Portfolio();
        portfolio.setTotalValue(125000.0);
        portfolio.setCashBalance(25000.0);
        portfolio.setUnrealizedPnL(3500.0);
        portfolio.setRealizedPnL(12000.0);
        portfolio.setWinRate(0.68);
        portfolio.setSharpeRatio(1.45);
        portfolio.setMaxDrawdown(0.12);
        portfolio.setTimestamp(System.currentTimeMillis());
        
        List<Position> positions = new ArrayList<>();
        Position btc = new Position();
        btc.setSymbol("BTCUSDT");
        btc.setSide("LONG");
        btc.setQuantity(1.5);
        btc.setEntryPrice(50000.0);
        btc.setCurrentPrice(52000.0);
        btc.setUnrealizedPnL(3000.0);
        btc.setPnlPercent(4.0);
        positions.add(btc);
        
        Position eth = new Position();
        eth.setSymbol("ETHUSDT");
        eth.setSide("LONG");
        eth.setQuantity(10.0);
        eth.setEntryPrice(2700.0);
        eth.setCurrentPrice(2800.0);
        eth.setUnrealizedPnL(1000.0);
        eth.setPnlPercent(3.7);
        positions.add(eth);
        
        portfolio.setPositions(positions);
        
        Map<String, Double> allocation = new HashMap<>();
        allocation.put("BTCUSDT", 0.60);
        allocation.put("ETHUSDT", 0.25);
        allocation.put("CASH", 0.15);
        portfolio.setAllocation(allocation);
        
        return portfolio;
    }
}
