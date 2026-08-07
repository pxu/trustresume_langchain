Backend Engineer with experience in Go, Python, and distributed systems. Skilled in building event-driven pipelines processing high-volume workloads, operating Kafka-based data exchange infrastructure, and managing production services on Kubernetes and PostgreSQL. Proven track record of reducing on-call burden through improved alerting and operational practices.

## Senior Backend Engineer, Northwind Logistics (2021–2026)
- Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day; cut p99 latency from 900 ms to 120 ms by replacing synchronous fan-out with an SQS buffer
- Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- Owned the CI/CD platform for 40 engineers: GitHub Actions runners on autoscaling EC2 and a shared caching layer that cut mean build time from 11 minutes to 4
- Wrote Terraform modules and a blue/green deploy pipeline for the billing service migration onto ECS Fargate, plus the rollback runbook the on-call team still uses
- Led a team of five engineers through two performance cycles; rebuilt the on-call rotation after a quarter averaging six pages/week, reducing pages to roughly one per week

## Technical Skills
- Languages: Go, Python
- Infrastructure & Orchestration: Kubernetes, Docker, Kafka, AWS (Lambda, EventBridge, SQS, ECS Fargate)
- Data Stores: PostgreSQL, Redis
- Monitoring & Alerting: Datadog (consumer-lag alerting), on-call rotation management
- Other: Terraform, GitHub Actions CI/CD, Schema Registry (Avro)