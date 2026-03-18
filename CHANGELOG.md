# AI Trading System - Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [2.3.0] - 2026-03-18

### Added

- **Concept Engine v2.0**: Knowledge layer finanziario con FAISS + sentence-transformers
  - Ricerca semantica ibrida (semantic + keyword)
  - 45+ concetti finanziari (trading, risk, market, DeFi, crypto, economics)
  - Estrazione concetti da news e testi
  - Sentiment analysis integrato
- Nuovo modulo `concept_engine.py` per NLP finanziario
- Librerie: faiss-cpu, sentence-transformers aggiunte ai requirements

### Fixed

- Dashboard Python (porta 8050): Corretto errore Series.to_dict() senza parametri
- Dashboard Python: Corretto parametro HiddenMarkovRegimeDetector (n_states → n_regimes)
- AI Assistant Streamlit: Sostituito use_container_width deprecato con width='stretch'
- Embedding manager: Corretto bug model_name non definito

### Changed

- Aggiornato versione a 2.3.0
- Migliorata compatibilità con pandas e hmmlearn

---

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
