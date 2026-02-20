# ✅ 5-Question Decision Engine - COMPLETED

> **Status**: Fully implemented in `decision_engine.py` (lines 1130-1650)
> **Last Updated**: 2026-02-20

---

## Phase 1: 5-Question Methods ✅
- [x] 1. `answer_what()` - What to buy/sell (ML + technical) → Line 1133
- [x] 2. `answer_why()` - Reason Score (0.6*Macro + 0.4*Sentiment) → Line 1200
- [x] 3. `answer_how_much()` - Position sizing (max_pos * reason_score) → Line 1254
- [x] 4. `answer_when()` - Monte Carlo timing score → Line 1292
- [x] 5. `answer_risk()` - Risk checks (VaR/CVaR, limits) → Line 1330

## Phase 2: Integration ✅
- [x] 6. `unified_decision()` method combining all 5 questions → Line 1418
- [x] 7. Macro Score fetching from external APIs → Integrated in `answer_why()`
- [x] 8. VaR/CVaR integration with risk engine → Integrated in `answer_risk()`

## Phase 3: Feedback Loop ✅
- [x] 9. Decision logging for feedback loop → Implemented via logger
- [x] 10. Config weights (0.6*Macro + 0.4*Sentiment) → Line 1238

## Phase 4: Testing ✅
- [x] 11. Test the new decision flow → `test_hedge_fund_features.py`
- [x] 12. Verify all components work together → `TestFiveQuestionDecisionEngine` class

---

## Implementation Details

### Method Signatures

```python
def answer_what(self, symbol: str, df: pd.DataFrame = None) -> Dict
def answer_why(self, symbol: str, df: pd.DataFrame = None) -> Dict
def answer_how_much(self, symbol: str, why_score: float, current_price: float = None) -> Dict
def answer_when(self, symbol: str, df: pd.DataFrame = None) -> Dict
def answer_risk(self, symbol: str, action: str, position_size: float, current_price: float, when_score: float = 0.5) -> Dict
def unified_decision(self, symbol: str) -> Dict
def generate_signals_5q(self, symbols: List[str] = None) -> List[TradingSignal]
```

### Return Values

Each method returns a dictionary with specific keys:

| Method | Key Fields |
|--------|------------|
| `answer_what()` | action, what_score, ml_direction, technical_direction |
| `answer_why()` | why_score, macro_score, sentiment_score, reason |
| `answer_how_much()` | how_much_score, position_size, position_value |
| `answer_when()` | when_score, probability_up, confidence, var_95 |
| `answer_risk()` | risk_score, passed, reason, var_95, cvar_95 |
| `unified_decision()` | Complete decision with all 5 answers |

