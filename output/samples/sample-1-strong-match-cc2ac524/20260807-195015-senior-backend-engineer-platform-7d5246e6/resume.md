Senior Backend Engineer with 5+ years of experience building high-volume, event-driven systems at scale in logistics. Expert in Python and Go, with deep hands-on ownership of order pipelines processing 40M+ events/day on AWS Lambda. Proven track record improving build and deploy tooling, leading on-call rotations, and mentoring engineering teams. Core stack: AWS, Kafka, Terraform, Docker, Kubernetes, Postgres, CI/CD, and Datadog.

## Senior Backend Engineer — Northwind Logistics (2021–2026)
- Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day, cutting p99 latency from 900 ms to 120 ms by replacing synchronous fan-out with an SQS buffer
- Owned the CI/CD platform for 40 engineers: GitHub Actions runners on autoscaling EC2 with a shared caching layer that reduced mean build time from 11 minutes to 4 minutes
- Authored Terraform modules and a blue/green deploy pipeline for the billing service migration onto ECS Fargate, including the rollback runbook used by the on-call team
- Introduced Apache Kafka as the backbone for cross-team data exchange — 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- Led on-call leadership initiative that rebuilt the on-call rotation, reducing weekly pages from six to approximately one
- Mentored a team of five engineers through two performance cycles, providing technical guidance and career development support
- Daily production stack: Python, Go, Postgres, Redis, Docker, and Kubernetes

## Technical Skills
- Languages: Python, Go
- Cloud & Infrastructure: AWS (Lambda, EventBridge, SQS, ECS Fargate, EC2), Terraform, Docker, Kubernetes
- Data & Messaging: Kafka (Avro/Schema Registry), Postgres, Redis
- CI/CD & Tooling: GitHub Actions, blue/green deployments, build and deploy tooling
- Observability: Datadog (consumer-lag alerting, monitoring dashboards)