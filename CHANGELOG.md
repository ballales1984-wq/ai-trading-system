# AI Trading System - Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- GitHub Projects board for issue tracking
- CONTRIBUTING.md for community contributions
- CODE_OF_CONDUCT.md for community guidelines
- 5-minute video tutorial (placeholder)
- Quick start guide with Docker deployment
- Discord community server

### Changed

- Improved README with screenshots and quick start guide
- Enhanced documentation structure
- Added deployment badges to README

### Fixed

- Deployment configuration issues
- Documentation inconsistencies

## [2.2.0] - 2026-03-16

### Added

- **Emergency Stop API**: Complete emergency stop functionality with `/api/orders/emergency-stop`, `/api/orders/emergency-resume`, and status endpoint
- **Demo Mode Support**: Full demo mode across all API endpoints (portfolio, market, orders)
- **Performance Metrics**: Dynamic daily performance calculations
- **ML Training**: Trained ML models for BTC, ETH, SOL price prediction
- **Financial Analysis**: Advanced finance modules integration

### Changed

- **Monte Carlo Simulations**: Changed from fixed seeds (42) to system time for variability
- **Risk Engine**: Improved risk metrics calculation
- **HMM Detection**: Fixed rolling window bug in regime detection
- **Frontend**: Optimized build with code splitting warnings addressed

### Fixed

- Monte Carlo random seed issues (deterministic simulations now variable)
- Risk metrics using fake data - now uses dynamic calculations
- HMM rolling window numpy array bug
- Performance metrics dynamic calculation

### Security

- Enhanced security middleware
- RBAC improvements
- Rate limiter enhancements

## [2.1.0] - 2026-03-07

### Added

- Futures trading support (Bybit testnet)
- Cross-exchange arbitrage detection
- Options trading module
- Multi-asset portfolio rebalancing

### Changed

- Improved risk management algorithms
- Enhanced execution engine with smart routing
- Updated frontend with dark mode and responsive design

### Fixed

- Docker build issues
- API endpoint optimizations
- Security vulnerabilities

## [2.0.0] - 2026-02-20

### Added

- React frontend with TypeScript
- Multi-broker support (Binance, Bybit, Paper Trading)
- Monte Carlo simulation (5-level)
- HMM regime detection
- Sentiment analysis integration

### Changed

- Complete architecture redesign
- Performance optimizations
- Test coverage improvements

### Fixed

- Database connection issues
- API endpoint bugs
- Security vulnerabilities

## [1.0.0] - 2025-12-15

### Added

- Basic trading system
- Single broker support
- Simple strategy engine
- Dashboard interface

### Changed

- Initial project structure
- Basic documentation

### Fixed

- Initial bug fixes
- Performance issues

---

## Template for Future Releases

### [X.X.X] - YYYY-MM-DD

#### Added

- Feature 1
- Feature 2

#### Changed

- Improvement 1
- Improvement 2

#### Deprecated

- Feature 1 (replaced by Feature 2)

#### Removed

- Feature 1 (no longer supported)

#### Fixed

- Bug 1
- Bug 2

#### Security

- Security issue 1
- Security issue 2
