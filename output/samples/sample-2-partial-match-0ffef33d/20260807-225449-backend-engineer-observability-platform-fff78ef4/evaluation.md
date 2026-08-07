# Evaluation

- **Verdict:** DID NOT PASS
- **Job title:** Backend Engineer, Observability Platform
- **Gate:** Trust ≥ 90 and ATS ≥ 85
- **Iterations run:** 3 (cap 3; 4 draft(s) total)
- **Model(s):** global.anthropic.claude-opus-4-6-v1

## Scores (iteration 3)

| Metric | Score | Threshold | Met |
|---|---|---|---|
| Trust | 94.4 | 90 | yes |
| ATS | 52.6 | 85 | **no** |

**Why it failed:** ATS score 53 (needs >= 85).

_Hit the rewrite cap without passing — this is the last draft, exported with its real scores rather than discarded (the loop keeps the final iteration; it does not search for the best-scoring one, unlike the trustresume original)._

## Iteration history

| Iteration | Trust | ATS | Passed |
|---|---|---|---|
| 0 | 82 | 74 | no |
| 1 | 97 | 47 | no |
| 2 | 97 | 58 | no |
| 3 (exported) | 94 | 53 | no |

## Trust Harness

- Claims extracted: 18
- Supported fraction: 0.89
- Flagged as unsupported: 0

- **PARTIALLY_SUPPORTED** (ACHIEVEMENT) — Rebuilt the on-call rotation after a quarter averaging six pages per week; reduced pages to roughly one per week through improved runbooks and operational processes
- **PARTIALLY_SUPPORTED** (SKILL) — Observability: Datadog (consumer-lag alerting, metrics dashboards)
- **SUPPORTED** (SKILL) — Backend Engineer with production experience in Go, Python, Kafka, Kubernetes, and PostgreSQL
- **SUPPORTED** (ACHIEVEMENT) — Building event-driven distributed systems processing 40M+ events/day
- **SUPPORTED** (ACHIEVEMENT) — Introducing Kafka-based data pipelines with schema enforcement and consumer-lag alerting
- **SUPPORTED** (ACHIEVEMENT) — Dramatically improving on-call reliability
- **SUPPORTED** (EXPERIENCE) — Senior Backend Engineer, Northwind Logistics (2021–2026)
- **SUPPORTED** (ACHIEVEMENT) — Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day
- **SUPPORTED** (ACHIEVEMENT) — Cut p99 latency from 900 ms to 120 ms by replacing synchronous fan-out with an SQS buffer
- **SUPPORTED** (ACHIEVEMENT) — Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- **SUPPORTED** (ACHIEVEMENT) — Owned CI/CD platform for 40 engineers: GitHub Actions runners on autoscaling EC2 with a shared caching layer that cut mean build time from 11 minutes to 4
- **SUPPORTED** (ACHIEVEMENT) — Wrote Terraform modules and blue/green deploy pipeline for billing service migration onto ECS Fargate, plus the rollback runbook the on-call team still uses
- **SUPPORTED** (EXPERIENCE) — Led a team of five engineers through two performance cycles
- **SUPPORTED** (SKILL) — Languages: Go, Python
- **SUPPORTED** (SKILL) — Infrastructure & Orchestration: Kubernetes, Docker, Terraform, AWS (Lambda, EventBridge, ECS Fargate, SQS, EC2)
- **SUPPORTED** (SKILL) — Data & Messaging: Apache Kafka (Avro, Schema Registry), PostgreSQL, Redis
- **SUPPORTED** (SKILL) — Observability: on-call rotation management
- **SUPPORTED** (OTHER) — Passionate about observability and operational excellence in high-throughput environments

## ATS keyword coverage

- Matched (10): Backend Engineer, Observability, Go, Python, Distributed systems, Kafka, Kubernetes, PostgreSQL, Metrics, On-call
- Missing (9): Telemetry, Time-series data, OpenTelemetry, gRPC, Prometheus, Grafana, SLOs, Traces, Ingestion

## Rewrite feedback

Deterministic, built from the gate's own gap (`orchestration/feedback.py`) — no extra LLM call.

```
Soften these claims to match what the evidence actually supports:
- Rebuilt the on-call rotation after a quarter averaging six pages per week; reduced pages to roughly one per week through improved runbooks and operational processes
- Observability: Datadog (consumer-lag alerting, metrics dashboards)
Where the evidence genuinely supports them, incorporate these target keywords the draft is missing: Telemetry, Time-series data, OpenTelemetry, gRPC, Prometheus, Grafana, SLOs, Traces, Ingestion
```

## Cost & latency

- 10 LLM calls · 28,936 tokens (18,353 in / 10,583 out) · $1.0690
- 120.9s wall clock
- Slowest step: score_trust (23.9s)
