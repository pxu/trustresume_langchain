Senior Backend Engineer with 5+ years of experience designing and operating high-scale event-driven systems and platform infrastructure. Proven ability to own CI/CD platform strategy serving 40 engineers, architect cross-team data exchange systems, and lead a team of five engineers through performance cycles. Skilled in Python, Go, AWS, Kafka, Kubernetes, and Terraform.

## Senior Backend Engineer, Northwind Logistics (2021–2026)
- Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day; cut p99 latency from 900ms to 120ms by replacing synchronous fan-out with an SQS buffer
- Owned the CI/CD platform serving 40 engineers: GitHub Actions runners on autoscaling EC2 with a shared caching layer that reduced mean build time from 11 minutes to 4 minutes
- Introduced Apache Kafka as the cross-team data exchange backbone—14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog—establishing shared systems architecture standards
- Led a team of five engineers through two performance cycles; rebuilt the on-call rotation, reducing pages from six per week to approximately one per week
- Wrote Terraform modules and a blue/green deploy pipeline for the billing service migration onto ECS Fargate, plus the rollback runbook still used by the on-call team

## Technical Skills
- Languages: Python, Go
- Infrastructure & Cloud: AWS (Lambda, EventBridge, SQS, ECS Fargate, EC2), Docker, Kubernetes, Terraform
- Data & Messaging: Apache Kafka, Avro/Schema Registry, PostgreSQL, Redis
- Observability & CI/CD: Datadog, GitHub Actions, blue/green deployments