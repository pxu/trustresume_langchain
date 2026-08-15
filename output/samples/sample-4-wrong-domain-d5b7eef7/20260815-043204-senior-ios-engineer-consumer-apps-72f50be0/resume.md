Senior Backend Engineer with 5 years of experience designing high-throughput, event-driven systems and leading engineering teams. Proven ability to ship production services processing tens of millions of events daily, optimize latency and reliability, and own CI/CD infrastructure at scale.

## Senior Backend Engineer — Northwind Logistics (2021–2026)
- Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day; cut p99 latency from 900 ms to 120 ms by replacing synchronous fan-out with an SQS buffer
- Owned the CI/CD platform for 40 engineers: GitHub Actions runners on autoscaling EC2 with a shared caching layer that reduced mean build time from 11 minutes to 4 minutes
- Wrote Terraform modules and blue/green deploy pipeline for billing service migration onto ECS Fargate, plus the rollback runbook still used by the on-call team
- Led a team of five engineers through two performance cycles; rebuilt on-call rotation reducing pages from six per week to roughly one per week
- Introduced Apache Kafka as backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog

## Technical Skills
- Languages: Python, Go
- Infrastructure: AWS (Lambda, EventBridge, SQS, ECS Fargate, EC2), Terraform, Docker, Kubernetes, GitHub Actions
- Data & Messaging: Apache Kafka, Avro, PostgreSQL, Redis
- Observability: Datadog