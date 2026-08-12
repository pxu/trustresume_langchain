# Evaluation

**DID NOT PASS** — Trust 100/90 · ATS 42/85 · iteration 3 of 3
**Why it failed:** ATS score 42 (needs >= 85).
_Hit the rewrite cap without passing — this is the last draft, exported with its real scores rather than discarded._

## Trust Harness

- **SUPPORTED** (SKILL) — Backend Engineer with experience in Go, Python, and distributed systems
- **SUPPORTED** (SKILL) — Skilled in building event-driven pipelines processing high-volume workloads
- **SUPPORTED** (SKILL) — Operating Kafka-based data exchange infrastructure
- **SUPPORTED** (SKILL) — Managing production services on Kubernetes and PostgreSQL
- **SUPPORTED** (ACHIEVEMENT) — Proven track record of reducing on-call burden through improved alerting and operational practices
- **SUPPORTED** (EXPERIENCE) — Senior Backend Engineer, Northwind Logistics (2021–2026)
- **SUPPORTED** (ACHIEVEMENT) — Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day
- **SUPPORTED** (ACHIEVEMENT) — Cut p99 latency from 900 ms to 120 ms by replacing synchronous fan-out with an SQS buffer
- **SUPPORTED** (ACHIEVEMENT) — Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- **SUPPORTED** (ACHIEVEMENT) — Owned the CI/CD platform for 40 engineers: GitHub Actions runners on autoscaling EC2 and a shared caching layer that cut mean build time from 11 minutes to 4
- **SUPPORTED** (ACHIEVEMENT) — Wrote Terraform modules and a blue/green deploy pipeline for the billing service migration onto ECS Fargate, plus the rollback runbook the on-call team still uses
- **SUPPORTED** (EXPERIENCE) — Led a team of five engineers through two performance cycles
- **SUPPORTED** (ACHIEVEMENT) — Rebuilt the on-call rotation after a quarter averaging six pages/week, reducing pages to roughly one per week
- **SUPPORTED** (SKILL) — Languages: Go, Python
- **SUPPORTED** (SKILL) — Infrastructure & Orchestration: Kubernetes, Docker, Kafka, AWS (Lambda, EventBridge, SQS, ECS Fargate)
- **SUPPORTED** (SKILL) — Data Stores: PostgreSQL, Redis
- **SUPPORTED** (SKILL) — Monitoring & Alerting: Datadog (consumer-lag alerting), on-call rotation management
- **SUPPORTED** (SKILL) — Other: Terraform, GitHub Actions CI/CD, Schema Registry (Avro)

## ATS keyword coverage

- Matched: Backend Engineer, Go, Python, Distributed systems, Kafka, Kubernetes, PostgreSQL, On-call
- Missing: Observability, Telemetry, Time-series data, OpenTelemetry, gRPC, Prometheus, Grafana, SLOs, Metrics, Traces, Ingestion

## Suggested improvements

```
Where the evidence genuinely supports them, incorporate these target keywords the draft is missing: Observability, Telemetry, Time-series data, OpenTelemetry, gRPC, Prometheus, Grafana, SLOs, Metrics, Traces, Ingestion
```

## Run cost

- 10 LLM calls · 29,202 tokens (18,334 in / 10,868 out)
- 123.1s wall clock · unknown (no price configured for this model)
