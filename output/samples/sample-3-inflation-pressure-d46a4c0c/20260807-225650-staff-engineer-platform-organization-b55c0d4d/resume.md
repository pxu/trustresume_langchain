Senior Backend Engineer with 5+ years of experience designing high-scale event-driven systems and owning platform infrastructure relied on by dozens of engineers. Proven ability to set technical direction for backend services, define architectural approaches for real-time data pipelines, and lead engineers through delivery and operational improvements. Seeking a Staff Engineer role in a platform organization to apply systems architecture expertise and engineering leadership at greater scale.

## Senior Backend Engineer, Northwind Logistics (2021–2026)
- Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day; cut p99 latency from 900ms to 120ms by replacing synchronous fan-out with an SQS buffer
- Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- Owned the CI/CD platform for 40 engineers—GitHub Actions runners on autoscaling EC2 with a shared caching layer that reduced mean build time from 11 minutes to 4 minutes
- Led a team of five engineers through two performance cycles; rebuilt the on-call rotation, reducing pages from six per week to approximately one per week
- Wrote Terraform modules and the blue/green deploy pipeline for the billing service migration onto ECS Fargate, plus the rollback runbook used by the on-call team
- Established technical direction for event-driven backend services and cross-team data infrastructure serving multiple downstream consumers

## Technical Skills
- Languages: Python, Go
- Infrastructure & Cloud: AWS (Lambda, EventBridge, SQS, ECS Fargate, EC2), Docker, Kubernetes, Terraform
- Data & Messaging: Apache Kafka (Avro, Schema Registry), PostgreSQL, Redis
- Practices: Event-driven architecture, high-scale systems design, CI/CD pipeline ownership, blue/green deployments