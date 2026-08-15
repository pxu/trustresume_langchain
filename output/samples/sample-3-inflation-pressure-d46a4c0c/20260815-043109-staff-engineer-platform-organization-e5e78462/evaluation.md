# Evaluation

- **Verdict:** DID NOT PASS
- **Job title:** Staff Engineer, Platform Organization
- **Gate:** Trust ≥ 90 and ATS ≥ 85
- **Iterations run:** 1 (cap 1; 2 draft(s) total)
- **Model(s):** global.anthropic.claude-opus-4-6-v1

## Scores (iteration 0)

| Metric | Score | Threshold | Met |
|---|---|---|---|
| Trust | 83.3 | 90 | **no** |
| ATS | 41.7 | 85 | **no** |

**Why it failed:** Trust score 83 (needs >= 90); ATS score 42 (needs >= 85).

_None of this run's drafts passed the gate — this is the best-scoring one across all iterations (highest ATS, since none passed to rank by), exported with its real scores rather than discarded._

## Iteration history

| Iteration | Trust | ATS | Passed |
|---|---|---|---|
| 0 (exported) | 83 | 42 | no |
| 1 | 97 | 25 | no |

## Trust Harness

- Claims extracted: 21
- Supported fraction: 0.81
- Flagged as unsupported: 3

- **UNSUPPORTED** (ACHIEVEMENT) — Establishing the backend technical direction for order fulfillment systems
- **UNSUPPORTED** (ACHIEVEMENT) — Defining the architectural roadmap for platform-wide event streaming
- **UNSUPPORTED** (ACHIEVEMENT) — Setting engineering standards across the organization
- **PARTIALLY_SUPPORTED** (ACHIEVEMENT) — Owned the company-wide CI/CD platform strategy for 40 engineers: GitHub Actions on autoscaling EC2 with shared caching, cutting mean build time from 11 minutes to 4 minutes
- **SUPPORTED** (EXPERIENCE) — 5+ years of experience designing high-scale, event-driven systems
- **SUPPORTED** (ACHIEVEMENT) — Processing tens of millions of events per day
- **SUPPORTED** (EXPERIENCE) — Owning CI/CD infrastructure for 40 engineers
- **SUPPORTED** (EXPERIENCE) — Leading teams through performance cycles
- **SUPPORTED** (SKILL) — Skilled in Python, Go, Kafka, and AWS services
- **SUPPORTED** (EXPERIENCE) — Senior Backend Engineer at Northwind Logistics, 2021–2026
- **SUPPORTED** (ACHIEVEMENT) — Architected and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day
- **SUPPORTED** (ACHIEVEMENT) — Reduced p99 latency from 900ms to 120ms by redesigning synchronous fan-out into an asynchronous SQS buffer architecture
- **SUPPORTED** (ACHIEVEMENT) — Introduced Apache Kafka as the backbone for cross-team data exchange—14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- **SUPPORTED** (EXPERIENCE) — Led a team of 5 engineers through two performance cycles
- **SUPPORTED** (ACHIEVEMENT) — Rebuilt on-call rotation reducing weekly pages from 6 to approximately 1
- **SUPPORTED** (ACHIEVEMENT) — Authored Terraform modules and blue/green deploy pipeline for billing service migration to ECS Fargate, plus rollback runbook used by on-call team
- **SUPPORTED** (SKILL) — Skilled in Kubernetes
- **SUPPORTED** (SKILL) — Skilled in Docker
- **SUPPORTED** (SKILL) — Skilled in PostgreSQL and Redis
- **SUPPORTED** (SKILL) — Skilled in Terraform
- **SUPPORTED** (SKILL) — Skilled in Datadog

## ATS keyword coverage

- Matched (5): platform strategy, technical direction, architectural roadmap, systems architecture, high-scale systems
- Missing (7): Staff Engineer, backend engineering, billions of events, engineering leadership, engineering managers, board-level communication, platform organization

## Rewrite feedback

Deterministic, built from the gate's own gap (`orchestration/feedback.py`) — no extra LLM call.

```
Remove or rephrase these claims — they are not supported by the candidate's evidence:
- Establishing the backend technical direction for order fulfillment systems (ACHIEVEMENT)
- Defining the architectural roadmap for platform-wide event streaming (ACHIEVEMENT)
- Setting engineering standards across the organization (ACHIEVEMENT)
Soften these claims to match what the evidence actually supports:
- Owned the company-wide CI/CD platform strategy for 40 engineers: GitHub Actions on autoscaling EC2 with shared caching, cutting mean build time from 11 minutes to 4 minutes
Where the evidence genuinely supports them, incorporate these target keywords the draft is missing: Staff Engineer, backend engineering, billions of events, engineering leadership, engineering managers, board-level communication, platform organization
```

## Cost & latency

- 6 LLM calls · 16,247 tokens (10,320 in / 5,927 out) · $0.5993
- 67.0s wall clock
- Slowest step: score_trust (23.1s)
