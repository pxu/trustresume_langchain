# Evaluation

- **Verdict:** PASSED
- **Job title:** Senior Backend Engineer - Platform
- **Gate:** Trust ≥ 90 and ATS ≥ 85
- **Iterations run:** 1 (cap 1; 2 draft(s) total)
- **Model(s):** global.anthropic.claude-opus-4-6-v1

## Scores (iteration 0)

| Metric | Score | Threshold | Met |
|---|---|---|---|
| Trust | 93.3 | 90 | yes |
| ATS | 95.2 | 85 | yes |

## Iteration history

| Iteration | Trust | ATS | Passed |
|---|---|---|---|
| 0 (exported) | 93 | 95 | yes |
| 1 | 100 | 90 | yes |

## Trust Harness

- Claims extracted: 15
- Supported fraction: 0.87
- Flagged as unsupported: 0

- **PARTIALLY_SUPPORTED** (EXPERIENCE) — Mentored a team of five engineers through two performance cycles, providing technical guidance and career development support
- **PARTIALLY_SUPPORTED** (EXPERIENCE) — Mentoring engineering teams
- **SUPPORTED** (EXPERIENCE) — Senior Backend Engineer with 5+ years of experience
- **SUPPORTED** (EXPERIENCE) — Building high-volume, event-driven systems at scale in logistics
- **SUPPORTED** (SKILL) — Expert in Python and Go
- **SUPPORTED** (EXPERIENCE) — Deep hands-on ownership of order pipelines processing 40M+ events/day on AWS Lambda
- **SUPPORTED** (ACHIEVEMENT) — Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day
- **SUPPORTED** (ACHIEVEMENT) — Cutting p99 latency from 900 ms to 120 ms by replacing synchronous fan-out with an SQS buffer
- **SUPPORTED** (ACHIEVEMENT) — Owned the CI/CD platform for 40 engineers: GitHub Actions runners on autoscaling EC2 with a shared caching layer that reduced mean build time from 11 minutes to 4 minutes
- **SUPPORTED** (ACHIEVEMENT) — Authored Terraform modules and a blue/green deploy pipeline for the billing service migration onto ECS Fargate, including the rollback runbook used by the on-call team
- **SUPPORTED** (ACHIEVEMENT) — Introduced Apache Kafka as the backbone for cross-team data exchange — 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- **SUPPORTED** (ACHIEVEMENT) — Led on-call leadership initiative that rebuilt the on-call rotation, reducing weekly pages from six to approximately one
- **SUPPORTED** (SKILL) — Daily production stack: Python, Go, Postgres, Redis, Docker, and Kubernetes
- **SUPPORTED** (EXPERIENCE) — Leading on-call rotations
- **SUPPORTED** (SKILL) — Core stack includes AWS, Kafka, Terraform, Docker, Kubernetes, Postgres, CI/CD, and Datadog

## ATS keyword coverage

- Matched (20): Python, AWS Lambda, AWS, Kafka, Terraform, CI/CD, Docker, Kubernetes, Postgres, Go, Datadog, on-call leadership, backend engineer, platform, logistics, high-volume, event-driven systems at scale, order pipeline, build and deploy tooling, mentoring
- Missing (1): event-driven architecture

## Cost & latency

- 6 LLM calls · 15,243 tokens (10,295 in / 4,948 out) · $0.5255
- 56.6s wall clock
- Slowest step: score_trust (15.7s)
