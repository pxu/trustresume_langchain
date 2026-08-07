MSAI 699 Capstone Project

Instructor: Dr. Gamini Bulumulle

Week 5: System Integration and Real-World Constraints: Deployment Strategy Report

Peng Fei Xu

University of the Cumberlands

Submitted to the University of the Cumberlands
in Partial Fulfillment of the Requirements of the Degree of
Master of Science in Artificial Intelligence

[Submission Date]

TrustResume: Evidence-Based Resume Generation Using RAG and Multi-Agent AI

---

## Abstract

This report evaluates the deployment approach for TrustResume, an evidence-based resume
generation system combining RAG, multi-agent orchestration, and a Trust Harness that scores
generated claims against retrieved evidence. It documents a deployment already implemented in the
project repository: a FastAPI backend containerized with Docker, a Streamlit frontend as a
separate service, and a provider-agnostic LLM layer supporting AWS Bedrock, OpenAI, and Google
Gemini. Before any deployment work began, agent orchestration was rebuilt on LangChain and
LangGraph in place of an earlier pydantic-ai prototype; since that prototype was never deployed,
this is the project's first deployment, built on the new stack rather than a migration of an
existing one. The report contrasts cloud and edge deployment across the pipeline's stages, finding
the system already a hybrid (retrieval at the edge, generation in the cloud), reviews the security
and fairness mechanisms already present, and argues the deployment is feasible at low
infrastructure cost, with a clear path to scale further if user demand grows.

## 1. Introduction

Deploying an AI system safely requires decisions beyond model accuracy: where computation runs,
how it is exposed to callers, and what harms it could introduce. TrustResume is a pipeline, not one
model: ingestion, embedding, hybrid retrieval, and five LLM-backed agents behind a quality loop.
Before deployment work began, agent orchestration was ported from pydantic-ai to **LangChain** and
**LangGraph**; Section 3 explains why, and why that should be read as the basis for this project's
first deployment rather than a change to an existing one.

## 2. Cloud vs. Edge Deployment

Embedding (FastEmbed) and retrieval (Chroma and SQLite/FTS5) are both edge-capable: they run
in-process with no server, so they execute inside the API container with no external database.
Generation, five to nine sequential LLM calls per run, is the one stage that requires the cloud,
since self-hosting a model comparable to Claude, GPT-4o, or Gemini is infeasible at this project's
scale; it runs through AWS Bedrock, OpenAI, or Gemini, chosen at deploy time via `LLMConfig`.

The result is a hybrid deployment. Candidate resume and career data, once embedded and stored,
never leaves the machine running the API container; only the minimum text a given agent call needs
(a job posting, retrieved evidence snippets) crosses the network to the LLM provider, matching a
constraint from the project's own Week 1 proposal. An offline `test` provider also lets the full
pipeline run with zero network access for development and CI, the closest thing to a fully edge
deployment, at the cost of not generating real content. Pinning generation to a cloud API means
inheriting that provider's availability and rate limits: `with_structured_retry` (`agents/base.py`)
retries each structured-output call up to three times so a multi-step run does not lose all
completed agent calls to one transient provider throttle.

## 3. API Integration

The TrustResume backend exposes RESTful APIs through FastAPI. FastAPI was selected instead of
Flask because of its native support for data validation through Pydantic (reusing the project's
own `models/` schemas directly as request and response bodies), automatic OpenAPI documentation
generation, and asynchronous request handling for I/O-heavy routes such as document upload.

Key endpoints include:

- **Documents**: `GET/POST /api/documents` (list/add already-extracted text), `POST
  /api/documents/upload` (ingest a raw file upload, parsed server-side), `DELETE
  /api/documents/{document_id}`
- **Jobs**: `POST/GET /api/jobs` (create/list), `GET/PUT/DELETE /api/jobs/{job_id}` (inspect,
  re-extract, remove), `POST /api/jobs/{job_id}/documents/upload` and `GET
  /api/jobs/{job_id}/documents` (documents scoped to one job)
- **Retrieval and generation**: `POST /api/search` (standalone retrieval, returning what the RAG
  pipeline would retrieve for a query without paying for a full generation), `POST /api/generate`
  and `POST /api/jobs/{job_id}/generate` (run the full pipeline against a raw posting or a
  persisted job)
- **Resumes**: `GET /api/jobs/{job_id}/resumes` (list past generations), `GET
  /api/resumes/{resume_id}` (detail), `GET /api/resumes/{resume_id}/pdf` and `/markdown` (export)
- **Liveness**: `GET /api/health` and `GET /api/ping` (provider connectivity check)

The Streamlit frontend communicates with the backend only through HTTP requests via a thin REST
client, never importing the backend package directly, which enables a clean separation of
presentation and business logic. Containerized deployment through Docker Compose allows both
services to be built, deployed, and scaled independently, with the frontend container depending on
the backend's health check rather than being coupled to its process.

Underneath this API, the agent and orchestration layer was originally prototyped on pydantic-ai,
before any part of the system was deployed, deliberately kept framework-light so control flow
stayed visible while the pipeline's behavior was still being worked out. Once that behavior was
stable, the project switched to LangChain (`with_structured_output` per agent) and LangGraph (the
orchestrator rebuilt as a `StateGraph`), the standard tools for a multi-step, multi-provider LLM
pipeline. LangChain's provider integrations are what let `model_factory.py` swap Bedrock, OpenAI,
or Gemini behind one config object, so the provider used at deploy time is a configuration choice
rather than a code change. Since the pydantic-ai prototype was never deployed, this switch has no
deployment impact to measure against; it is a design decision made in preparation for the
deployment described here, and LangGraph's `StateGraph` is what the orchestrator runs on today.

## 4. Ethical Concerns in Deployment

**Security.** Prompts built from untrusted text (a job posting, an uploaded document) are
delimited and marked as data, never instructions, guarding against prompt injection. API keys are
excluded from logging and kept out of version control. Authentication is the clearest gap: the
deployed API operates on one hardcoded demo user for every route, fine for a personal or demo
deployment but not for real multi-tenant traffic.

**Fairness and bias.** The Trust Harness, checking each claim against retrieved evidence, doubles
as a fairness mechanism: an ungrounded generator would favor candidates whose backgrounds happen to
embellish well under the model's own biases. ATS keyword-coverage scoring carries a residual bias
risk of its own (rewarding phrasing that mirrors the posting), so its score is treated as feedback
to the candidate, not a claim about a real employer's system.

**Privacy.** Every stored record and vector carries a user id, filtered server-side on every
search; that is the real isolation boundary, not just a convention. Documents stay local, and the
only third party that ever sees candidate text is the configured LLM provider, only for what a
given call needs.

## 5. Deployment Feasibility

The proposed deployment is technically feasible because:

- FastAPI has low infrastructure requirements and can run on a single virtual machine.
- ChromaDB and SQLite eliminate the need for separate database servers.
- Docker Compose simplifies deployment and environment consistency.
- LLM providers such as AWS Bedrock, OpenAI, and Gemini eliminate the need to host large
  foundation models.
- The architecture can scale incrementally by moving the API container to Kubernetes or
  cloud-managed services when user demand increases.

## 6. Conclusion

TrustResume's deployment, FastAPI behind Docker Compose, a LangChain/LangGraph agent layer chosen
deliberately as the basis for the project's first deployment, and an embedded local data layer, is
already a hybrid cloud/edge split driven by where each stage's computation can run. The security
and fairness mechanisms already in the codebase address the risks specific to this pipeline, and
the deployment is feasible at low infrastructure cost, with an incremental path (moving the API
container to Kubernetes or a cloud-managed service) if user demand ever grows beyond a single
host. The clearest remaining gap is authentication and rate limiting; both are easy to add since
every store call is already user-scoped, but neither exists yet.

## AI Disclosure Statement

In August 2026, during Week 5 of this project, an AI coding assistant (Claude) was used to review
the project's deployment configuration and to draft and revise this report. All architectural
claims describe work already completed in the project repository in earlier milestones; no new
deployment infrastructure was built for this report. All findings and conclusions were reviewed
and confirmed by the author.
