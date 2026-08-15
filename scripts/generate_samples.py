"""Generate the sample résumés + evaluations in ``output/samples/``.

Runs **one candidate against four job postings** through the real pipeline, so
the committed artifacts show the quality gate passing *and* failing in each of
the ways it can fail. A system that only ever shows you passing output isn't
demonstrating a quality gate, it's hiding one.

1. **strong-match** — the posting asks for what the evidence documents.
   Passes.
2. **partial-match** — genuine overlap, genuine gaps (an observability role for
   a generalist backend engineer). Fails on both axes; useful for watching the
   rewrite loop try and run out of room.
3. **inflation-pressure** — a Staff-level posting asking for scope one step
   beyond the evidence: 15 years, teams of 20+, org-wide strategy. This is the
   pressure that produces quiet exaggeration on a real résumé. The writer does
   stretch, and the Trust Harness catches it — the clearest demonstration of
   why generation and verification are separate steps.
4. **wrong-domain** — an iOS posting. The writer refuses to invent iOS
   experience, so Trust stays high and ATS goes to zero: the system fails the
   candidate rather than lying for them.

Throwaway script, like ``manual_rag_test.py`` — not imported, not tested, and
it uses an isolated temp DB/Chroma so it never touches the repo's real
``trustresume.db``/``chroma_data``. Artifacts *are* written to the repo's
``output/`` on purpose: looking at them is the entire reason to run this.

    TRUSTRESUME_LLM_PROVIDER=bedrock python scripts/generate_samples.py

Under ``TRUSTRESUME_LLM_PROVIDER=test`` all four fail identically (the offline
fake extracts no claims, so Trust is always 0) — deterministic, and useless for
this comparison. The script says so rather than pretending.

Output is LLM-generated, so re-running will not reproduce these numbers
exactly. The *shape* of each outcome is what the scenarios are designed to be
stable about, not the scores.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from trustresume.api.app_service import build_default_app
from trustresume.api.model_factory import LLMConfig
from trustresume.models import DocumentType, WorkflowState

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output" / "samples"

INDEX_START = "<!-- INDEX:START -->"
INDEX_END = "<!-- INDEX:END -->"

# One candidate's real-looking career evidence. Deliberately concrete —
# numbers, product names, team sizes — because the Trust Harness can only
# support a claim as far as the evidence is specific.
EVIDENCE = """
Senior Backend Engineer, Northwind Logistics (2021-2026)

Designed and shipped an event-driven order pipeline on AWS Lambda and
EventBridge that processes 40 million events per day. Cut p99 latency from
900ms to 120ms by replacing synchronous fan-out with an SQS buffer.

Owned the CI/CD platform for 40 engineers: GitHub Actions runners on
autoscaling EC2 and a shared caching layer that cut mean build time from 11
minutes to 4.

Wrote the Terraform modules and the blue/green deploy pipeline for the billing
service migration onto ECS Fargate, plus the rollback runbook the on-call team
still uses.

Led a team of five engineers through two performance cycles. Rebuilt the
on-call rotation after a quarter averaging six pages a week; pages dropped to
roughly one a week.

Introduced Apache Kafka as the backbone for cross-team data exchange: 14
topics with schema-registry-enforced Avro contracts and consumer-lag alerting
in Datadog.

Python and Go day to day. Postgres, Redis, Docker, Kubernetes in production.
"""

# Asks for precisely what the evidence documents.
STRONG_MATCH_JOB = """
Senior Backend Engineer - Platform, Meridian Freight

We run a high-volume logistics platform on AWS and need someone who has
operated event-driven systems at scale.

Required: Python, AWS Lambda, event-driven architecture, Kafka, Terraform,
CI/CD, Docker, Kubernetes, Postgres.
Nice to have: Go, Datadog, on-call leadership experience.

You will own the order pipeline, improve build and deploy tooling for the
platform team, and mentor engineers.
"""

# Asks for a different discipline entirely. The candidate is strong — just not
# at this job.
WEAK_MATCH_JOB = """
Senior iOS Engineer - Consumer Apps, Lumen Media

We are building a video-first consumer app used by millions.

Required: Swift, SwiftUI, UIKit, Core Data, AVFoundation, XCTest, 5+ years
shipping native iOS apps to the App Store, Metal shader experience, and deep
knowledge of iOS memory management and battery profiling.
Nice to have: Objective-C maintenance, App Store review experience, ARKit.

You will own the video playback layer and the offline download experience.
"""

# Asks for adjacent-but-larger scope than the evidence supports: staff level,
# 15 years, org-wide ownership, teams of 20+. Every one of those is a small
# step beyond what the evidence documents — which is exactly the pressure that
# produces quiet inflation on a real résumé, and the pressure the Trust
# Harness exists to resist.
INFLATION_PRESSURE_JOB = """
Staff Engineer, Platform Organization - Atlas Freight

We need a Staff Engineer to set backend technical direction across the whole
platform organization.

Required: 15+ years of backend engineering; experience leading teams of 20 or
more engineers; ownership of an entire company-wide platform strategy; proven
record architecting systems handling billions of events per day; direct
management of multiple engineering managers; and board-level communication of
technical strategy.

You will define the multi-year architectural roadmap for the organization.
"""

# Overlaps meaningfully but not completely: the candidate has the backend and
# AWS depth, and no observability-vendor or gRPC specifics. A first draft
# should miss keywords and get one or more rewrites.
PARTIAL_MATCH_JOB = """
Backend Engineer, Observability Platform - Northstar Systems

Help us build the telemetry pipeline our whole engineering org depends on.

Required: Go or Python, distributed systems experience, Kafka, Kubernetes,
PostgreSQL, and hands-on work with high-cardinality time-series data.
Nice to have: OpenTelemetry instrumentation, gRPC, Prometheus, Grafana,
experience running an on-call rotation and defining SLOs.

You will own ingestion, storage, and query for metrics and traces.
"""

SCENARIOS = [
    ("1-strong-match", STRONG_MATCH_JOB, "passes - evidence covers the posting"),
    (
        "2-partial-match",
        PARTIAL_MATCH_JOB,
        "real overlap, real gaps - fails on both axes after using every rewrite",
    ),
    (
        "3-inflation-pressure",
        INFLATION_PRESSURE_JOB,
        "posting invites exaggeration; the harness flags what the writer stretched",
    ),
    (
        "4-wrong-domain",
        WEAK_MATCH_JOB,
        "fails on ATS - honest draft, but the evidence has none of these skills",
    ),
]


def _summarize(name: str, expectation: str, state: WorkflowState) -> None:
    """Print the outcome the way the artifacts render it.

    Reads ``final_*`` (the draft this run actually ships), not ``current_*``
    (the latest iteration) — the quality loop no longer stops on the first
    pass, so the two can differ (ADR-0016).
    """
    trust = state.final_trust
    ats = state.final_ats
    if trust is None or ats is None:
        print(f"  {name}: produced no scored draft", file=sys.stderr)
        return

    verdict = "PASSED" if state.final_passed else "DID NOT PASS"
    print(f"\n  [{name}] {expectation}")
    print(f"    result      {verdict}")
    print(f"    trust       {trust.score:.0f} / {state.gate.min_trust_score:.0f}")
    print(f"    ats         {ats.score:.0f} / {state.gate.min_ats_score:.0f}")
    print(f"    drafts      {len(state.drafts)} (cap {state.gate.max_iterations + 1})")
    if trust.claims:
        by_status: dict[str, int] = {}
        for claim in trust.claims:
            by_status[claim.status.value] = by_status.get(claim.status.value, 0) + 1
        print(f"    claims      {by_status}")
    if ats.missing_keywords:
        print(f"    missing     {', '.join(ats.missing_keywords[:6])}")
    if state.usage:
        usage = state.usage
        cost = f"${usage.cost_usd:.4f}" if usage.cost_usd is not None else "unpriced"
        print(
            f"    cost        {usage.llm_calls} calls · {usage.total_tokens:,} tokens · "
            f"{usage.total_duration_ms / 1000:.1f}s · {cost}"
        )


def write_index() -> int:
    """Regenerate the Results table in ``output/samples/README.md`` from disk.

    Derived from the committed ``evaluation.json`` files rather than
    hand-written, so the table cannot drift from the runs it describes — a
    stale index in a directory whose whole purpose is honest example output
    would be a small lie in the most inconvenient place. Each scenario
    directory may hold more than one timestamped run (re-running this script
    adds another); the table shows the most recent one per scenario.
    """
    readme = OUTPUT_DIR / "README.md"
    if not readme.is_file():
        print(f"{readme} not found", file=sys.stderr)
        return 1

    rows = [
        "| Scenario | Result | Trust /90 | ATS /85 | Drafts | Claims flagged |",
        "|---|---|---|---|---|---|",
    ]
    for scenario_dir in sorted(d for d in OUTPUT_DIR.iterdir() if d.is_dir()):
        runs = sorted(scenario_dir.glob("*/evaluation.json"))
        if not runs:
            continue
        latest = runs[-1]
        payload = json.loads(latest.read_text(encoding="utf-8"))
        scenario_name = scenario_dir.name.rsplit("-", 1)[0].removeprefix("sample-")
        flagged = sum(1 for c in payload["trust"]["claims"] if c["status"] != "SUPPORTED")
        rows.append(
            f"| [{scenario_name}]({scenario_dir.name}/{latest.parent.name}/evaluation.md) "
            f"| **{'PASS' if payload['passed'] else 'FAIL'}** "
            f"| {payload['trust']['score']:.0f} | {payload['ats']['score']:.0f} "
            f"| {payload['iteration'] + 1} | {flagged} |"
        )

    table = "\n".join(rows)
    content = readme.read_text(encoding="utf-8")
    if INDEX_START in content and INDEX_END in content:
        head, rest = content.split(INDEX_START, 1)
        _, tail = rest.split(INDEX_END, 1)
        content = f"{head}{INDEX_START}\n{table}\n{INDEX_END}{tail}"
    else:
        content = content.rstrip() + f"\n\n## Results\n\n{INDEX_START}\n{table}\n{INDEX_END}\n"
    readme.write_text(content, encoding="utf-8")
    print(f"indexed {len(rows) - 2} scenarios into {readme.relative_to(REPO_ROOT)}")
    return 0


def main() -> None:
    if "--reindex" in sys.argv:
        raise SystemExit(write_index())

    config = LLMConfig.from_env()
    if config.provider == "test":
        print(
            "The offline 'test' provider extracts no claims, so Trust is always 0 and\n"
            "both scenarios fail identically — nothing to compare. Re-run with a real\n"
            "provider, e.g.:\n\n"
            "    TRUSTRESUME_LLM_PROVIDER=bedrock python scripts/generate_samples.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    tmp_dir = tempfile.mkdtemp(prefix="trustresume_samples_")
    print(f"[setup] provider={config.provider} model={config.model_name()}", file=sys.stderr)
    print(f"[setup] isolated stores in {tmp_dir}", file=sys.stderr)
    print(f"[setup] artifacts -> {OUTPUT_DIR}", file=sys.stderr)
    try:
        with build_default_app(
            db_path=str(Path(tmp_dir) / "samples.db"),
            chroma_path=str(Path(tmp_dir) / "samples_chroma"),
            llm_config=config,
            output_dir=str(OUTPUT_DIR),
        ) as app:
            for name, posting, expectation in SCENARIOS:
                # A separate user per scenario, so each gets its own artifact
                # folder and neither can retrieve the other's evidence.
                user_id = f"sample-{name}"
                app.ensure_user(name, user_id=user_id)
                app.add_document(
                    user_id=user_id,
                    filename="career_evidence.txt",
                    text=EVIDENCE,
                    document_type=DocumentType.RESUME,
                )
                print(f"\n[run] {name} …", file=sys.stderr)
                state = app.generate(user_id=user_id, job_posting=posting)
                _summarize(name, expectation, state)

        print(f"\nArtifacts written under {OUTPUT_DIR}:")
        for path in sorted(OUTPUT_DIR.rglob("evaluation.md")):
            print(f"  {path.relative_to(REPO_ROOT)}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    write_index()


if __name__ == "__main__":
    main()
