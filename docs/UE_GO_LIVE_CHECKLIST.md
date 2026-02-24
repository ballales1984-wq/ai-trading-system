# UE Go-Live Checklist (Trading App)

Last updated: 2026-02-23
Scope: EU launch readiness for AI Trading System
Status: Actionable checklist (engineering + product + legal ops)

## Decision Summary

Go/no-go: `YES, with controls`.

You can launch in the EU if these controls are implemented before public rollout.  
If a control marked `BLOCKER` is missing, release must be delayed.

## 0-30 Days (Blockers)

- [ ] `BLOCKER` Define service perimeter in plain language:
  software tool vs investment advice vs execution service.
- [ ] `BLOCKER` Publish legal pages and link them from all entry points:
  Terms, Privacy, Risk Disclosure, Contact.
- [ ] `BLOCKER` Add explicit consent capture with versioning:
  accepted_at, document_version, user_id, IP/device evidence.
- [ ] `BLOCKER` Add product copy guardrails:
  no guaranteed returns, no misleading performance claims, capital-at-risk warnings.
- [ ] `BLOCKER` Data protection baseline:
  legal basis map, retention schedule, DSAR process, breach escalation path.
- [ ] `BLOCKER` Security baseline:
  MFA for admin, secret rotation policy, audit log for auth/order/risk actions.

## 31-60 Days (Pre-Beta Hardening)

- [ ] Implement jurisdiction gating:
  allowed countries, blocked flows, clear user eligibility logic.
- [ ] Add operational resilience runbook:
  incidents, fallback modes, RTO/RPO targets, on-call ownership.
- [ ] Add model/transparency controls for AI features:
  what is automated, what is user-driven, limitations and confidence caveats.
- [ ] Add compliance evidence pack:
  policy docs, change logs, control owners, review cadence.

## 61-90 Days (Launch Readiness)

- [ ] External legal review in target EU jurisdictions.
- [ ] Final regulatory decision memo:
  whether licensing/registration obligations apply to final business model.
- [ ] Marketing review workflow:
  pre-publication legal/compliance sign-off for ads, testimonials, social posts.
- [ ] Post-launch monitoring:
  KPI + risk dashboard for incidents, complaints, and policy breaches.

## Release Gate (Must Be True)

- [ ] All `BLOCKER` items completed.
- [ ] Legal texts are published and versioned.
- [ ] Consent records are queryable and exportable.
- [ ] Incident and escalation contacts are assigned.
- [ ] Final counsel sign-off is documented.

## Ownership Template

- Product Owner: scope, user flows, copy controls.
- Engineering Lead: technical controls, logs, consent evidence.
- Security Lead: IAM, secrets, auditability, incident response.
- Legal/Compliance Counsel: perimeter, disclosures, jurisdiction position.

## Notes

- This checklist is an engineering execution aid, not legal advice.
- Use this together with `docs/GO_TO_MARKET_COMPLIANCE_PLAN.md` and `LEGAL.md`.
