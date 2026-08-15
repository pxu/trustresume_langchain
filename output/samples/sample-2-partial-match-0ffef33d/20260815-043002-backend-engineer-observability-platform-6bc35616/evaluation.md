# Evaluation

- **Verdict:** DID NOT PASS
- **Job title:** Backend Engineer, Observability Platform
- **Gate:** Trust ≥ 90 and ATS ≥ 85
- **Iterations run:** 1 (cap 1; 2 draft(s) total)
- **Model(s):** global.anthropic.claude-opus-4-6-v1

## Scores (iteration 0)

| Metric | Score | Threshold | Met |
|---|---|---|---|
| Trust | 81.2 | 90 | **no** |
| ATS | 63.2 | 85 | **no** |

**Why it failed:** Trust score 81 (needs >= 90); ATS score 63 (needs >= 85).

_None of this run's drafts passed the gate — this is the best-scoring one across all iterations (highest ATS, since none passed to rank by), exported with its real scores rather than discarded._

## Iteration history

| Iteration | Trust | ATS | Passed |
|---|---|---|---|
| 0 (exported) | 81 | 63 | no |
| 1 | 97 | 47 | no |

## Trust Harness

- Claims extracted: 16
- Supported fraction: 0.62
- Flagged as unsupported: 0

- **PARTIALLY_SUPPORTED** (EXPERIENCE) — Introducing Kafka-based telemetry and data exchange infrastructure
- **PARTIALLY_SUPPORTED** (ACHIEVEMENT) — Experienced with on-call rotation ownership, defining alerting strategies, and driving reliability improvements that reduced incident pages by 85%
- **PARTIALLY_SUPPORTED** (ACHIEVEMENT) — Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting, enabling reliable high-throughput ingestion across distributed services
- **PARTIALLY_SUPPORTED** (ACHIEVEMENT) — Rebuilt the on-call rotation and alerting strategy after a quarter averaging six pages/week; reduced pages to roughly one per week through improved SLOs and monitoring
- **PARTIALLY_SUPPORTED** (EXPERIENCE) — Led a team of five engineers through two performance cycles, mentoring on distributed systems best practices and operational excellence
- **PARTIALLY_SUPPORTED** (SKILL) — Observability & Reliability: Datadog, consumer-lag alerting, on-call rotation management, SLO-driven incident reduction
- **SUPPORTED** (EXPERIENCE) — 5+ years of experience building high-throughput distributed systems in Go and Python
- **SUPPORTED** (ACHIEVEMENT) — Designing event-driven pipelines processing 40M+ events/day
- **SUPPORTED** (EXPERIENCE) — Operating services on Kubernetes and PostgreSQL in production
- **SUPPORTED** (ACHIEVEMENT) — Designed and shipped an event-driven order pipeline processing 40 million events per day on AWS, cutting p99 latency from 900ms to 120ms by replacing synchronous fan-out with an asynchronous buffer
- **SUPPORTED** (EXPERIENCE) — Operated production services on Kubernetes, PostgreSQL, Redis, and Docker; wrote Go and Python daily for backend services and tooling
- **SUPPORTED** (ACHIEVEMENT) — Owned CI/CD platform for 40 engineers: autoscaling GitHub Actions runners with shared caching layer that cut mean build time from 11 minutes to 4 minutes
- **SUPPORTED** (ACHIEVEMENT) — Authored Terraform modules and blue/green deploy pipeline for billing service migration onto ECS Fargate, plus the rollback runbook used by the on-call team
- **SUPPORTED** (SKILL) — Languages: Go, Python
- **SUPPORTED** (SKILL) — Infrastructure & Data: Kafka, Kubernetes, PostgreSQL, Redis, Docker, AWS (Lambda, EventBridge, SQS, ECS Fargate)
- **SUPPORTED** (SKILL) — CI/CD & IaC: GitHub Actions, Terraform, blue/green deployments

## ATS keyword coverage

- Matched (12): Backend Engineer, Observability, Telemetry, Go, Python, Distributed systems, Kafka, Kubernetes, PostgreSQL, SLOs, Ingestion, On-call
- Missing (7): Time-series data, OpenTelemetry, gRPC, Prometheus, Grafana, Metrics, Traces

## Rewrite feedback

Deterministic, built from the gate's own gap (`orchestration/feedback.py`) — no extra LLM call.

```
Soften these claims to match what the evidence actually supports:
- Introducing Kafka-based telemetry and data exchange infrastructure
- Experienced with on-call rotation ownership, defining alerting strategies, and driving reliability improvements that reduced incident pages by 85%
- Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting, enabling reliable high-throughput ingestion across distributed services
- Rebuilt the on-call rotation and alerting strategy after a quarter averaging six pages/week; reduced pages to roughly one per week through improved SLOs and monitoring
- Led a team of five engineers through two performance cycles, mentoring on distributed systems best practices and operational excellence
- Observability & Reliability: Datadog, consumer-lag alerting, on-call rotation management, SLO-driven incident reduction
Where the evidence genuinely supports them, incorporate these target keywords the draft is missing: Time-series data, OpenTelemetry, gRPC, Prometheus, Grafana, Metrics, Traces
```

## Cost & latency

- 6 LLM calls · 15,862 tokens (10,420 in / 5,442 out) · $0.5645
- 61.6s wall clock
- Slowest step: score_trust (20.1s)
