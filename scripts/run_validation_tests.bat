@echo off
REM ============================================
REM Run Automated Testing Framework
REM ============================================

echo.
echo ========================================
echo AI Trading System - Validation Tests
echo ========================================
echo.

REM Run all validation levels
python -m tests.automated_testing_framework

echo.
echo ========================================
echo Results saved to: test_report_full_validation.json
echo ========================================
echo.

pause
