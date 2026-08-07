# Evaluation

**DID NOT PASS** — Trust 85/90 · ATS 58/85 · iteration 3 of 3
**Why it failed:** Trust score 85 (needs >= 90); ATS score 58 (needs >= 85).
_Hit the rewrite cap without passing — this is the last draft, exported with its real scores rather than discarded._

## Trust Harness

- **UNSUPPORTED** (OTHER) — Seeking to apply backend engineering expertise and architectural thinking at the Staff Engineer level within a platform organization
- **UNSUPPORTED** (ACHIEVEMENT) — Strengthening systems architecture across the organization
- **UNSUPPORTED** (EXPERIENCE) — Drove architectural decisions for backend engineering infrastructure used across multiple teams, contributing to platform-level technical direction
- **PARTIALLY_SUPPORTED** (SKILL) — Datadog (consumer-lag alerting, latency monitoring)
- **SUPPORTED** (EXPERIENCE) — Senior Backend Engineer with 5+ years of experience
- **SUPPORTED** (EXPERIENCE) — Designing and operating high-scale systems processing tens of millions of events per day
- **SUPPORTED** (SKILL) — Event-driven pipelines experience
- **SUPPORTED** (EXPERIENCE) — CI/CD platform ownership
- **SUPPORTED** (EXPERIENCE) — Cross-team data infrastructure experience
- **SUPPORTED** (EXPERIENCE) — Led a team of five engineers
- **SUPPORTED** (ACHIEVEMENT) — Driven platform improvements serving 40+ engineers
- **SUPPORTED** (ACHIEVEMENT) — Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day
- **SUPPORTED** (ACHIEVEMENT) — Cut p99 latency from 900ms to 120ms by replacing synchronous fan-out with an SQS buffer
- **SUPPORTED** (ACHIEVEMENT) — Owned the CI/CD platform serving 40 engineers: architected GitHub Actions runners on autoscaling EC2 with a shared caching layer, reducing mean build time from 11 minutes to 4 minutes
- **SUPPORTED** (ACHIEVEMENT) — Introduced Apache Kafka as the backbone for cross-team data exchange — 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- **SUPPORTED** (ACHIEVEMENT) — Led a team of five engineers through two performance cycles; rebuilt the on-call rotation, reducing pages from six per week to approximately one per week
- **SUPPORTED** (ACHIEVEMENT) — Authored Terraform modules and a blue/green deploy pipeline for the billing service migration onto ECS Fargate, including the rollback runbook still used by the on-call team
- **SUPPORTED** (SKILL) — Languages: Python, Go
- **SUPPORTED** (SKILL) — AWS (Lambda, EventBridge, SQS, ECS Fargate, EC2)
- **SUPPORTED** (SKILL) — Docker, Kubernetes
- **SUPPORTED** (SKILL) — Terraform
- **SUPPORTED** (SKILL) — Apache Kafka (Avro, Schema Registry)
- **SUPPORTED** (SKILL) — PostgreSQL, Redis
- **SUPPORTED** (SKILL) — GitHub Actions

**2 claim(s) flagged as unsupported factual assertions — the ones to fix first.**

## ATS keyword coverage

- Matched: Staff Engineer, backend engineering, technical direction, systems architecture, high-scale systems, engineering leadership, platform organization
- Missing: platform strategy, architectural roadmap, billions of events, engineering managers, board-level communication

## Suggested improvements

```
Remove or rephrase these claims — they are not supported by the candidate's evidence:
- Strengthening systems architecture across the organization (ACHIEVEMENT)
- Drove architectural decisions for backend engineering infrastructure used across multiple teams, contributing to platform-level technical direction (EXPERIENCE)
Soften these claims to match what the evidence actually supports:
- Datadog (consumer-lag alerting, latency monitoring)
Where the evidence genuinely supports them, incorporate these target keywords the draft is missing: platform strategy, architectural roadmap, billions of events, engineering managers, board-level communication
```

## Run cost

- 10 LLM calls · 28,503 tokens (17,073 in / 11,430 out)
- 132.1s wall clock · unknown (no price configured for this model)
