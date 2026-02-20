# TODO: 5-Question Decision Engine Integration

## Phase 1: Add 5-Question Methods
- [ ] 1. answer_what() - What to buy/sell (ML + technical)
- [ ] 2. answer_why() - Reason Score (0.6*Macro + 0.4*Sentiment)  
- [ ] 3. answer_how_much() - Position sizing (max_pos * reason_score)
- [ ] 4. answer_when() - Monte Carlo timing score
- [ ] 5. answer_risk() - Risk checks (VaR/CVaR, limits)

## Phase 2: Integration
- [ ] 6. Create unified_decision() method combining all 5 questions
- [ ] 7. Add Macro Score fetching from external APIs
- [ ] 8. Integrate with risk engine for VaR/CVaR

## Phase 3: Feedback Loop
- [ ] 9. Add decision logging for feedback loop
- [ ] 10. Update config with new weights (0.6*Macro + 0.4*Sentiment)

## Phase 4: Testing
- [ ] 11. Test the new decision flow
- [ ] 12. Verify all components work together

