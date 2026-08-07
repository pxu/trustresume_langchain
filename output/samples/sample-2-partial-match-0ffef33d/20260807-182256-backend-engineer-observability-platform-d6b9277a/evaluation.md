# Evaluation

**DID NOT PASS** — Trust 84/90 · ATS 45/85 · iteration 3 of 3
**Why it failed:** Trust score 84 (needs >= 90); ATS score 45 (needs >= 85).
_Hit the rewrite cap without passing — this is the last draft, exported with its real scores rather than discarded._

## Trust Harness

- **PARTIALLY_SUPPORTED** (EXPERIENCE) — Backend Engineer with 5+ years of experience
- **PARTIALLY_SUPPORTED** (SKILL) — Managing production Kubernetes and PostgreSQL environments
- **PARTIALLY_SUPPORTED** (ACHIEVEMENT) — Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog for monitoring ingestion health
- **PARTIALLY_SUPPORTED** (ACHIEVEMENT) — Rebuilt the on-call rotation after a quarter averaging six pages per week; reduced pages to roughly one per week through improved alerting and operational processes
- **PARTIALLY_SUPPORTED** (SKILL) — Monitoring & Observability: Datadog (consumer-lag alerting, dashboards)
- **PARTIALLY_SUPPORTED** (OTHER) — Focus on reliability and reducing operational burden
- **SUPPORTED** (SKILL) — Building high-throughput distributed systems in Go and Python
- **SUPPORTED** (ACHIEVEMENT) — Designing event-driven pipelines processing tens of millions of events daily
- **SUPPORTED** (SKILL) — Operating Kafka-based architectures with schema enforcement
- **SUPPORTED** (SKILL) — Experienced with on-call rotation management and monitoring via Datadog
- **SUPPORTED** (EXPERIENCE) — Senior Backend Engineer, Northwind Logistics (2021–2026)
- **SUPPORTED** (ACHIEVEMENT) — Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day
- **SUPPORTED** (ACHIEVEMENT) — Cut p99 latency from 900ms to 120ms by replacing synchronous fan-out with an SQS buffer
- **SUPPORTED** (ACHIEVEMENT) — Owned the CI/CD platform for 40 engineers: GitHub Actions runners on autoscaling EC2 with a shared caching layer that cut mean build time from 11 minutes to 4
- **SUPPORTED** (ACHIEVEMENT) — Wrote Terraform modules and the blue/green deploy pipeline for the billing service migration onto ECS Fargate, plus the rollback runbook the on-call team still uses
- **SUPPORTED** (EXPERIENCE) — Led a team of five engineers through two performance cycles
- **SUPPORTED** (SKILL) — Languages: Go, Python
- **SUPPORTED** (SKILL) — Infrastructure & Data: Kafka, Kubernetes, PostgreSQL, Redis, Docker, AWS (Lambda, EventBridge, SQS, ECS Fargate)
- **SUPPORTED** (SKILL) — Practices: Distributed systems design, event-driven architectures, CI/CD, Infrastructure as Code (Terraform)

## ATS keyword coverage

- Matched: Backend Engineer, Observability, Go, Python, distributed systems, Kafka, Kubernetes, PostgreSQL, on-call, ingestion
- Missing: telemetry pipeline, time-series data, high-cardinality, metrics, traces, OpenTelemetry, gRPC, Prometheus, Grafana, SLOs, storage, query

## Suggested improvements

```
Soften these claims to match what the evidence actually supports:
- Backend Engineer with 5+ years of experience
- Managing production Kubernetes and PostgreSQL environments
- Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog for monitoring ingestion health
- Rebuilt the on-call rotation after a quarter averaging six pages per week; reduced pages to roughly one per week through improved alerting and operational processes
- Monitoring & Observability: Datadog (consumer-lag alerting, dashboards)
- Focus on reliability and reducing operational burden
Where the evidence genuinely supports them, incorporate these target keywords the draft is missing: telemetry pipeline, time-series data, high-cardinality, metrics, traces, OpenTelemetry, gRPC, Prometheus, Grafana, SLOs, storage, query
```

## Run cost

- 10 LLM calls · 28,389 tokens (17,269 in / 11,120 out)
- 128.6s wall clock · unknown (no price configured for this model)
