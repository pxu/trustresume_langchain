Senior Backend Engineer with 5+ years of experience designing and operating high-scale systems processing tens of millions of events per day. Proven track record in systems architecture, including event-driven pipelines, CI/CD platform ownership, and cross-team data infrastructure. Experienced in engineering leadership, having led a team of five engineers and driven platform improvements serving 40+ engineers. Seeking to apply backend engineering expertise and architectural thinking at the Staff Engineer level within a platform organization.

## Senior Backend Engineer — Northwind Logistics (2021–2026)
- Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day, cutting p99 latency from 900ms to 120ms by replacing synchronous fan-out with an SQS buffer
- Owned the CI/CD platform serving 40 engineers: architected GitHub Actions runners on autoscaling EC2 with a shared caching layer, reducing mean build time from 11 minutes to 4 minutes
- Introduced Apache Kafka as the backbone for cross-team data exchange — 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog, strengthening systems architecture across the organization
- Led a team of five engineers through two performance cycles; rebuilt the on-call rotation, reducing pages from six per week to approximately one per week
- Authored Terraform modules and a blue/green deploy pipeline for the billing service migration onto ECS Fargate, including the rollback runbook still used by the on-call team
- Drove architectural decisions for backend engineering infrastructure used across multiple teams, contributing to platform-level technical direction

## Technical Skills
- Languages: Python, Go
- Infrastructure & Orchestration: AWS (Lambda, EventBridge, SQS, ECS Fargate, EC2), Docker, Kubernetes, Terraform
- Data & Messaging: Apache Kafka (Avro, Schema Registry), PostgreSQL, Redis
- CI/CD & Observability: GitHub Actions, Datadog (consumer-lag alerting, latency monitoring)
- Architecture Patterns: Event-driven systems, blue/green deployments, high-scale backend systems