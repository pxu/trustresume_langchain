Backend Engineer with production experience in Go, Python, Kafka, Kubernetes, and PostgreSQL. Proven track record building event-driven distributed systems processing 40M+ events/day, introducing Kafka-based data pipelines with schema enforcement and consumer-lag alerting, and dramatically improving on-call reliability. Passionate about observability and operational excellence in high-throughput environments.

## Senior Backend Engineer, Northwind Logistics (2021–2026)
- Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day; cut p99 latency from 900 ms to 120 ms by replacing synchronous fan-out with an SQS buffer
- Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog for observability into distributed data flows
- Rebuilt the on-call rotation after a quarter averaging six pages per week; reduced pages to roughly one per week through improved runbooks and operational processes
- Owned CI/CD platform for 40 engineers: GitHub Actions runners on autoscaling EC2 with a shared caching layer that cut mean build time from 11 minutes to 4
- Wrote Terraform modules and blue/green deploy pipeline for billing service migration onto ECS Fargate, plus the rollback runbook the on-call team still uses
- Led a team of five engineers through two performance cycles

## Technical Skills
- Languages: Go, Python
- Infrastructure & Orchestration: Kubernetes, Docker, Terraform, AWS (Lambda, EventBridge, ECS Fargate, SQS, EC2)
- Data & Messaging: Apache Kafka (Avro, Schema Registry), PostgreSQL, Redis
- Observability: Datadog (consumer-lag alerting, metrics dashboards), on-call rotation management
- Practices: Distributed systems design, event-driven architecture, CI/CD, blue/green deployments