# Go-To-Market Compliance Plan (Dev -> Production)

Last updated: 2026-02-23
Scope: AI Trading System (marketing + dashboard + API)
Status: Draft implementation plan for development stage.

## 1) What we are now (development mode)

- Frontend app on Vercel (React dashboard).
- Backend API on local machine, exposed via ngrok.
- Marketing landing active and connected to waitlist.

## 2) Regulatory baseline to design for (official sources)

Checked on: 2026-02-23

### US (high impact if you market/advice to US users)

- SEC adviser framework / registration obligations:
  - https://www.sec.gov/divisions/investment/iaregulation/regia.htm
- SEC Marketing Rule focus (testimonials, endorsements, disclosures):
  - https://www.sec.gov/examinations-focused-new-investment-adviser-marketing-rule
  - https://www.sec.gov/rules-regulations/staff-guidance/division-investment-management-frequently-asked-questions/marketing-compliance-frequently-asked-questions
- SEC Form CRS for retail relationships (if applicable to regulated model):
  - https://www.sec.gov/investment/form-crs-faq
- CFTC/NFA (if commodity/derivatives advice/execution falls in scope):
  - https://www.nfa.futures.org/registration-membership/who-has-to-register/index.html
  - https://www.nfa.futures.org/registration-membership/who-has-to-register/cta.html
- CFTC advisory against AI trading scam claims:
  - https://www.cftc.gov/PressRoom/PressReleases/8854-24

### EU/Italy (high impact if targeting EU/Italian users)

- MAR and social media investment recommendation requirements (ESMA):
  - https://www.esma.europa.eu/press-news/esma-news/requirements-when-posting-investments-recommendations-social-media
- GDPR official regulation text:
  - https://eur-lex.europa.eu/eli/reg/2016/679/oj
- AI Act official regulation text:
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng
- Italy market abuse/unauthorized service enforcement context (CONSOB):
  - https://www.consob.it/web/area-pubblica/dettaglio-news/-/asset_publisher/qjVSo44Lk1fI/content/comunicato-stampa-del-20-marzo-2025-abusivismo/10194

### Advertising and disclosure standards

- FTC .com disclosures / clear and conspicuous standard:
  - https://www.ftc.gov/business-guidance/resources/com-disclosures-how-make-effective-disclosures-digital-advertising
  - https://www.ftc.gov/business-guidance/resources/native-advertising-guide-businesses
- FTC endorsements guidance:
  - https://www.ftc.gov/business-guidance/resources/ftcs-endorsement-guides

## 3) Market references (for product/disclaimer patterns)

- eToro risk disclosure format:
  - https://www.etoro.com/customer-service/general-risk-disclosure/
- TradingView disclaimer structure:
  - https://www.tradingview.com/disclaimer/
- 3Commas legal disclaimer (software-only/no advice):
  - https://client.3commas.io/legal_disclaimer
- QuantConnect no-advice and algorithmic risk language:
  - https://www.quantconnect.com/terms/

Reference pages validated on: 2026-02-23

## 4) Product/legal architecture target

- Single source of API truth in production (no duplicated backend behavior).
- Marketing and dashboard separated by route and access control.
- Explicit onboarding flow: marketing -> account -> consent -> plan -> dashboard.
- Feature gating by subscription state and risk profile.
- Audit logs and immutable event trail for sensitive actions.

## 5) Implementation phases

### Phase A (now, development-safe)

- Add visible legal links and risk disclaimers to landing.
- Add legal placeholders: Terms, Privacy, Risk Disclosure.
- Add "no investment advice / no guaranteed returns" copy where claims appear.
- Keep kill-switch, audit logging, and access controls in active development.

### Phase B (before public beta)

- Authentication + email verification + password reset.
- Consent capture with versioning:
  - Terms accepted at timestamp/version
  - Privacy accepted at timestamp/version
  - Risk disclosure acknowledged at timestamp/version
- Plan/paywall wiring (demo/paper/pro).

### Phase C (before paid launch)

- Legal review by licensed counsel in target jurisdictions.
- Compliance decision tree:
  - software-only tool vs advisory service
  - allowed jurisdictions
  - investor category restrictions
- Monitoring + incident response + legal data retention policy.

## 6) Non-negotiable copy controls

- Never promise guaranteed returns.
- Always show capital-at-risk warning near CTA and performance widgets.
- If testimonials/endorsements exist, disclose paid relationship clearly.
- Separate education/demo content from live-trading execution claims.

## 7) Immediate TODO (this sprint)

- [x] Link landing/footer legal navigation to real legal routes.
- [x] Add risk warning section near signup CTA.
- [ ] Add legal acceptance checkbox in signup flow (dev placeholder).
- [x] Add dashboard top banner: "Educational and informational use; not investment advice."
- [ ] Add compliance notes in README for dev/staging/prod.

---

Important: this plan is an engineering compliance framework, not legal advice.
Final legal position must be validated by qualified counsel before launch.
