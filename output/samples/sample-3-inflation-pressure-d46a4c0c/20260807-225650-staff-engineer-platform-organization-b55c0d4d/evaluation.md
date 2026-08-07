# Evaluation

- **Verdict:** DID NOT PASS
- **Job title:** Staff Engineer, Platform Organization
- **Gate:** Trust ≥ 90 and ATS ≥ 85
- **Iterations run:** 3 (cap 3; 4 draft(s) total)
- **Model(s):** global.anthropic.claude-opus-4-6-v1

## Scores (iteration 3)

| Metric | Score | Threshold | Met |
|---|---|---|---|
| Trust | 92.1 | 90 | yes |
| ATS | 50.0 | 85 | **no** |

**Why it failed:** ATS score 50 (needs >= 85).

_Hit the rewrite cap without passing — this is the last draft, exported with its real scores rather than discarded (the loop keeps the final iteration; it does not search for the best-scoring one, unlike the trustresume original)._

## Iteration history

| Iteration | Trust | ATS | Passed |
|---|---|---|---|
| 0 | 92 | 50 | no |
| 1 | 100 | 50 | no |
| 2 | 85 | 67 | no |
| 3 (exported) | 92 | 50 | no |

## Trust Harness

- Claims extracted: 19
- Supported fraction: 0.84
- Flagged as unsupported: 0

- **PARTIALLY_SUPPORTED** (SKILL) — Proven ability to set technical direction for backend services
- **PARTIALLY_SUPPORTED** (SKILL) — Define architectural approaches for real-time data pipelines
- **PARTIALLY_SUPPORTED** (EXPERIENCE) — Established technical direction for event-driven backend services and cross-team data infrastructure serving multiple downstream consumers
- **SUPPORTED** (EXPERIENCE) — Senior Backend Engineer with 5+ years of experience
- **SUPPORTED** (SKILL) — Designing high-scale event-driven systems
- **SUPPORTED** (EXPERIENCE) — Owning platform infrastructure relied on by dozens of engineers
- **SUPPORTED** (EXPERIENCE) — Lead engineers through delivery and operational improvements
- **SUPPORTED** (EXPERIENCE) — Senior Backend Engineer, Northwind Logistics (2021–2026)
- **SUPPORTED** (ACHIEVEMENT) — Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day
- **SUPPORTED** (ACHIEVEMENT) — Cut p99 latency from 900ms to 120ms by replacing synchronous fan-out with an SQS buffer
- **SUPPORTED** (ACHIEVEMENT) — Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- **SUPPORTED** (ACHIEVEMENT) — Owned the CI/CD platform for 40 engineers—GitHub Actions runners on autoscaling EC2 with a shared caching layer that reduced mean build time from 11 minutes to 4 minutes
- **SUPPORTED** (EXPERIENCE) — Led a team of five engineers through two performance cycles
- **SUPPORTED** (ACHIEVEMENT) — Rebuilt the on-call rotation, reducing pages from six per week to approximately one per week
- **SUPPORTED** (ACHIEVEMENT) — Wrote Terraform modules and the blue/green deploy pipeline for the billing service migration onto ECS Fargate, plus the rollback runbook used by the on-call team
- **SUPPORTED** (SKILL) — Languages: Python, Go
- **SUPPORTED** (SKILL) — Infrastructure & Cloud: AWS (Lambda, EventBridge, SQS, ECS Fargate, EC2), Docker, Kubernetes, Terraform
- **SUPPORTED** (SKILL) — Data & Messaging: Apache Kafka (Avro, Schema Registry), PostgreSQL, Redis
- **SUPPORTED** (SKILL) — Practices: Event-driven architecture, high-scale systems design, CI/CD pipeline ownership, blue/green deployments

## ATS keyword coverage

- Matched (6): Staff Engineer, platform organization, technical direction, systems architecture, high-scale systems, engineering leadership
- Missing (6): backend engineering, architectural roadmap, platform strategy, engineering managers, board-level communication, billions of events per day

## Rewrite feedback

Deterministic, built from the gate's own gap (`orchestration/feedback.py`) — no extra LLM call.

```
Soften these claims to match what the evidence actually supports:
- Proven ability to set technical direction for backend services
- Define architectural approaches for real-time data pipelines
- Established technical direction for event-driven backend services and cross-team data infrastructure serving multiple downstream consumers
Where the evidence genuinely supports them, incorporate these target keywords the draft is missing: backend engineering, architectural roadmap, platform strategy, engineering managers, board-level communication, billions of events per day
```

## Cost & latency

- 10 LLM calls · 28,657 tokens (18,157 in / 10,500 out) · $1.0599
- 121.5s wall clock
- Slowest step: score_trust (22.7s)
