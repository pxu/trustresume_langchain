# Evaluation

- **Verdict:** PASSED
- **Job title:** Senior Backend Engineer - Platform
- **Gate:** Trust ≥ 90 and ATS ≥ 85
- **Iterations run:** 0 (cap 3; 1 draft(s) total)
- **Model(s):** global.anthropic.claude-opus-4-6-v1

## Scores (iteration 0)

| Metric | Score | Threshold | Met |
|---|---|---|---|
| Trust | 93.8 | 90 | yes |
| ATS | 90.0 | 85 | yes |

## Trust Harness

- Claims extracted: 16
- Supported fraction: 0.88
- Flagged as unsupported: 0

- **PARTIALLY_SUPPORTED** (EXPERIENCE) — Mentored junior and mid-level engineers on system design and operational excellence
- **PARTIALLY_SUPPORTED** (ACHIEVEMENT) — Rebuilt the on-call rotation after a quarter averaging 6 pages/week; reduced pages to ~1/week through improved alerting, runbooks, and architectural fixes
- **SUPPORTED** (EXPERIENCE) — Senior Backend Engineer with 5+ years of experience
- **SUPPORTED** (SKILL) — Building high-volume, event-driven platforms in Python and Go
- **SUPPORTED** (ACHIEVEMENT) — Owning order pipelines processing 40M+ events/day on AWS Lambda
- **SUPPORTED** (EXPERIENCE) — Leading CI/CD and deploy tooling initiatives for platform teams
- **SUPPORTED** (EXPERIENCE) — Mentoring engineers through performance cycles
- **SUPPORTED** (SKILL) — Deep expertise in Kafka, Terraform, Docker, Kubernetes, and Postgres
- **SUPPORTED** (ACHIEVEMENT) — On-call leadership experience that reduced incident pages by 80%
- **SUPPORTED** (ACHIEVEMENT) — Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day
- **SUPPORTED** (ACHIEVEMENT) — Cut p99 latency from 900 ms to 120 ms by replacing synchronous fan-out with an SQS buffer
- **SUPPORTED** (ACHIEVEMENT) — Owned the CI/CD platform for 40 engineers: GitHub Actions runners on autoscaling EC2 with a shared caching layer that reduced mean build time from 11 minutes to 4 minutes
- **SUPPORTED** (ACHIEVEMENT) — Authored Terraform modules and a blue/green deploy pipeline for the billing service migration onto ECS Fargate, including the rollback runbook used by the on-call team
- **SUPPORTED** (ACHIEVEMENT) — Introduced Apache Kafka as the backbone for cross-team data exchange — 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- **SUPPORTED** (EXPERIENCE) — Led a team of 5 engineers through two performance cycles
- **SUPPORTED** (SKILL) — Daily stack: Python, Go, Postgres, Redis, Docker, Kubernetes on AWS

## ATS keyword coverage

- Matched (18): Python, AWS Lambda, AWS, event-driven architecture, Kafka, Terraform, CI/CD, Docker, Kubernetes, Postgres, Go, Datadog, on-call leadership, high-volume, backend engineer, order pipeline, deploy tooling, mentoring
- Missing (2): logistics platform, platform engineering

## Cost & latency

- 4 LLM calls · 9,355 tokens (6,362 in / 2,993 out) · $0.3199
- 35.9s wall clock
- Slowest step: score_trust (17.3s)
