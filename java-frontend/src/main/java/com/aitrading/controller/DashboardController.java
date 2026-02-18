package com.aitrading.controller;

import com.aitrading.service.TradingApiService;
import com.aitrading.model.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * Web controller for trading dashboard pages
 */
@Controller
public class DashboardController {
    
    @Autowired
    private TradingApiService tradingService;
    
    /**
     * Main dashboard page
     */
    @GetMapping("/")
    public String dashboard(Model model) {
        // Get all data for dashboard
        List<String> symbols = tradingService.getSymbols();
        List<TradingSignal> signals = tradingService.getSignals();
        Portfolio portfolio = tradingService.getPortfolio();
        
        model.addAttribute("symbols", symbols);
        model.addAttribute("signals", signals);
        model.addAttribute("portfolio", portfolio);
        model.addAttribute("page", "dashboard");
        
        return "dashboard";
    }
    
    /**
     * Market overview page
     */
    @GetMapping("/market")
    public String market(Model model, @RequestParam(required = false, defaultValue = "BTCUSDT") String symbol) {
        MarketData marketData = tradingService.getMarketData(symbol);
        List<String> symbols = tradingService.getSymbols();
        
        model.addAttribute("selectedSymbol", symbol);
        model.addAttribute("symbols", symbols);
        model.addAttribute("marketData", marketData);
        model.addAttribute("page", "market");
        
        return "market";
    }
    
    /**
     * Signals page
     */
    @GetMapping("/signals")
    public String signals(Model model) {
        List<TradingSignal> signals = tradingService.getSignals();
        
        model.addAttribute("signals", signals);
        model.addAttribute("page", "signals");
        
        return "signals";
    }
    
    /**
     * Portfolio page
     */
    @GetMapping("/portfolio")
    public String portfolio(Model model) {
        Portfolio portfolio = tradingService.getPortfolio();
        
        model.addAttribute("portfolio", portfolio);
        model.addAttribute("page", "portfolio");
        
        return "portfolio";
    }
    
    /**
     * Trade execution page
     */
    @GetMapping("/trade")
    public String trade(Model model) {
        List<String> symbols = tradingService.getSymbols();
        
        model.addAttribute("symbols", symbols);
        model.addAttribute("page", "trade");
        
        return "trade";
    }
    
    /**
     * Execute trade API endpoint
     */
    @PostMapping("/api/execute-trade")
    @ResponseBody
    public Map<String, Object> executeTrade(@RequestParam String symbol, 
                                            @RequestParam String side, 
                                            @RequestParam double quantity) {
        return tradingService.executeTrade(symbol, side, quantity);
    }
    
    /**
     * API endpoint for real-time data (used by WebSocket)
     */
    @GetMapping("/api/market-data")
    @ResponseBody
    public MarketData getMarketData(@RequestParam String symbol) {
        return tradingService.getMarketData(symbol);
    }
    
    /**
     * API endpoint for signals
     */
    @GetMapping("/api/signals")
    @ResponseBody
    public List<TradingSignal> getSignals() {
        return tradingService.getSignals();
    }
    
    /**
     * API endpoint for portfolio
     */
    @GetMapping("/api/portfolio")
    @ResponseBody
    public Portfolio getPortfolio() {
        return tradingService.getPortfolio();
    }
}
