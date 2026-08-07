Backend Engineer with 5+ years of experience building high-throughput distributed systems in Go and Python. Proven track record designing event-driven pipelines processing tens of millions of events daily, operating Kafka-based architectures with schema enforcement, and managing production Kubernetes and PostgreSQL environments. Experienced with on-call rotation management and monitoring via Datadog, with a focus on reliability and reducing operational burden.

## Senior Backend Engineer, Northwind Logistics (2021–2026)
- Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day; cut p99 latency from 900ms to 120ms by replacing synchronous fan-out with an SQS buffer
- Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog for monitoring ingestion health
- Rebuilt the on-call rotation after a quarter averaging six pages per week; reduced pages to roughly one per week through improved alerting and operational processes
- Owned the CI/CD platform for 40 engineers: GitHub Actions runners on autoscaling EC2 with a shared caching layer that cut mean build time from 11 minutes to 4
- Wrote Terraform modules and the blue/green deploy pipeline for the billing service migration onto ECS Fargate, plus the rollback runbook the on-call team still uses
- Led a team of five engineers through two performance cycles

## Technical Skills
- Languages: Go, Python
- Infrastructure & Data: Kafka, Kubernetes, PostgreSQL, Redis, Docker, AWS (Lambda, EventBridge, SQS, ECS Fargate)
- Monitoring & Observability: Datadog (consumer-lag alerting, dashboards), on-call rotation management
- Practices: Distributed systems design, event-driven architectures, CI/CD, Infrastructure as Code (Terraform)