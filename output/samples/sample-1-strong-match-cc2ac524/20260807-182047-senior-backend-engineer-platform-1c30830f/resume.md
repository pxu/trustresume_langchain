Senior Backend Engineer with 5+ years of experience building high-volume, event-driven microservices on AWS. Proficient in Python and Go, with hands-on expertise in AWS Lambda, Kafka, Terraform, Docker, Kubernetes, and Postgres. Proven on-call leadership and track record of mentoring engineers, improving platform reliability, and owning CI/CD tooling for cross-functional teams in logistics environments.

## Senior Backend Engineer — Northwind Logistics (2021–2026)
- Designed and shipped an event-driven order pipeline on AWS Lambda and EventBridge processing 40 million events per day; cut p99 latency from 900 ms to 120 ms by replacing synchronous fan-out with an SQS buffer
- Introduced Apache Kafka as the backbone for cross-team data exchange: 14 topics with schema-registry-enforced Avro contracts and consumer-lag alerting in Datadog
- Owned the CI/CD platform for 40 engineers — GitHub Actions runners on autoscaling EC2 with a shared caching layer that reduced mean build time from 11 minutes to 4
- Wrote Terraform modules and a blue/green deploy pipeline for the billing service migration onto ECS Fargate, plus the rollback runbook the on-call team still uses
- Led a team of five engineers through two performance cycles, providing mentoring on system design and operational practices
- Provided on-call leadership by rebuilding the on-call rotation after a quarter averaging six pages/week; pages dropped to approximately one per week
- Daily work in Python and Go; production systems built on Postgres, Redis, Docker, and Kubernetes