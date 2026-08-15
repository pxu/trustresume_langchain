Backend Engineer with 5+ years of experience building high-throughput distributed systems in Go and Python. Proven track record designing event-driven pipelines processing 40M+ events/day, introducing Kafka-based telemetry and data exchange infrastructure, and operating services on Kubernetes and PostgreSQL in production. Experienced with on-call rotation ownership, defining alerting strategies, and driving reliability improvements that reduced incident pages by 85%.

## Senior Backend Engineer, Northwind Logistics (2021–2026)
- Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting, enabling reliable high-throughput ingestion across distributed services
- Designed and shipped an event-driven order pipeline processing 40 million events per day on AWS, cutting p99 latency from 900ms to 120ms by replacing synchronous fan-out with an asynchronous buffer
- Operated production services on Kubernetes, PostgreSQL, Redis, and Docker; wrote Go and Python daily for backend services and tooling
- Rebuilt the on-call rotation and alerting strategy after a quarter averaging six pages/week; reduced pages to roughly one per week through improved SLOs and monitoring
- Owned CI/CD platform for 40 engineers: autoscaling GitHub Actions runners with shared caching layer that cut mean build time from 11 minutes to 4 minutes
- Authored Terraform modules and blue/green deploy pipeline for billing service migration onto ECS Fargate, plus the rollback runbook used by the on-call team
- Led a team of five engineers through two performance cycles, mentoring on distributed systems best practices and operational excellence

## Technical Skills
- Languages: Go, Python
- Infrastructure & Data: Kafka, Kubernetes, PostgreSQL, Redis, Docker, AWS (Lambda, EventBridge, SQS, ECS Fargate)
- Observability & Reliability: Datadog, consumer-lag alerting, on-call rotation management, SLO-driven incident reduction
- CI/CD & IaC: GitHub Actions, Terraform, blue/green deployments