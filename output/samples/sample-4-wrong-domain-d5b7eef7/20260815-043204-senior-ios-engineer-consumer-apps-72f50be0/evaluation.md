# Evaluation

- **Verdict:** DID NOT PASS
- **Job title:** Senior iOS Engineer - Consumer Apps
- **Gate:** Trust ≥ 90 and ATS ≥ 85
- **Iterations run:** 1 (cap 1; 2 draft(s) total)
- **Model(s):** global.anthropic.claude-opus-4-6-v1

## Scores (iteration 1)

| Metric | Score | Threshold | Met |
|---|---|---|---|
| Trust | 100.0 | 90 | yes |
| ATS | 0.0 | 85 | **no** |

**Why it failed:** ATS score 0 (needs >= 85).

_None of this run's drafts passed the gate — this is the best-scoring one across all iterations (highest ATS, since none passed to rank by), exported with its real scores rather than discarded._

## Iteration history

| Iteration | Trust | ATS | Passed |
|---|---|---|---|
| 0 | 100 | 0 | no |
| 1 (exported) | 100 | 0 | no |

## Trust Harness

- Claims extracted: 16
- Supported fraction: 1.00
- Flagged as unsupported: 0

- **SUPPORTED** (EXPERIENCE) — Senior Backend Engineer with 5 years of experience
- **SUPPORTED** (SKILL) — Designing high-throughput, event-driven systems
- **SUPPORTED** (EXPERIENCE) — Leading engineering teams
- **SUPPORTED** (ACHIEVEMENT) — Ship production services processing tens of millions of events daily
- **SUPPORTED** (EXPERIENCE) — Own CI/CD infrastructure at scale
- **SUPPORTED** (ACHIEVEMENT) — Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day
- **SUPPORTED** (ACHIEVEMENT) — Cut p99 latency from 900 ms to 120 ms by replacing synchronous fan-out with an SQS buffer
- **SUPPORTED** (ACHIEVEMENT) — Owned the CI/CD platform for 40 engineers: GitHub Actions runners on autoscaling EC2 with a shared caching layer that reduced mean build time from 11 minutes to 4 minutes
- **SUPPORTED** (ACHIEVEMENT) — Wrote Terraform modules and blue/green deploy pipeline for billing service migration onto ECS Fargate, plus the rollback runbook still used by the on-call team
- **SUPPORTED** (EXPERIENCE) — Led a team of five engineers through two performance cycles
- **SUPPORTED** (ACHIEVEMENT) — Rebuilt on-call rotation reducing pages from six per week to roughly one per week
- **SUPPORTED** (ACHIEVEMENT) — Introduced Apache Kafka as backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- **SUPPORTED** (SKILL) — Languages: Python, Go
- **SUPPORTED** (SKILL) — Infrastructure: AWS (Lambda, EventBridge, SQS, ECS Fargate, EC2), Terraform, Docker, Kubernetes, GitHub Actions
- **SUPPORTED** (SKILL) — Data & Messaging: Apache Kafka, Avro, PostgreSQL, Redis
- **SUPPORTED** (SKILL) — Observability: Datadog

## ATS keyword coverage

- Matched (0): (none)
- Missing (17): Swift, SwiftUI, UIKit, Core Data, AVFoundation, XCTest, Metal, iOS, memory management, battery profiling, video playback, offline download, native iOS apps, App Store, consumer apps, ARKit, Objective-C

## Rewrite feedback

Deterministic, built from the gate's own gap (`orchestration/feedback.py`) — no extra LLM call.

```
Where the evidence genuinely supports them, incorporate these target keywords the draft is missing: Swift, SwiftUI, UIKit, Core Data, AVFoundation, XCTest, Metal, iOS, memory management, battery profiling, video playback, offline download, native iOS apps, App Store, consumer apps, ARKit, Objective-C
```

## Cost & latency

- 6 LLM calls · 14,860 tokens (10,102 in / 4,758 out) · $0.5084
- 54.9s wall clock
- Slowest step: score_trust (16.4s)
