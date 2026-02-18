package com.aitrading;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * AI Trading Dashboard - Spring Boot Application
 * 
 * Main entry point for the Java-based trading dashboard.
 * Connects to the Python backend APIs for trading data and signals.
 */
@SpringBootApplication
@EnableScheduling
public class AiTradingApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(AiTradingApplication.class, args);
    }
}
