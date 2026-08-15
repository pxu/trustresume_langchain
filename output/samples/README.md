# Sample output — one candidate, four job postings

Real output from the pipeline, committed so you can read it without running
anything. Everything here is **synthetic**: the candidate, the employers, and
all four postings are invented (see `scripts/generate_samples.py`). No real
résumé has ever been committed to this repository.

Each run directory holds five files:

| File | What it is |
|---|---|
| `resume.md` / `resume.pdf` | the generated résumé |
| `evaluation.md` | **start here** — verdict, every claim with its verdict, missing keywords, cost |
| `evaluation.json` | the same plus the audit trail: which evidence chunk supported which claim |
| `job.md` | the posting it was written against |

## Results

<!-- INDEX:START -->
| Scenario | Result | Trust /90 | ATS /85 | Drafts | Claims flagged |
|---|---|---|---|---|---|
| [1-strong-match](sample-1-strong-match-cc2ac524/20260815-042901-senior-backend-engineer-platform-27f60f59/evaluation.md) | **PASS** | 93 | 95 | 2 | 2 |
| [2-partial-match](sample-2-partial-match-0ffef33d/20260815-043002-backend-engineer-observability-platform-6bc35616/evaluation.md) | **FAIL** | 81 | 63 | 2 | 6 |
| [3-inflation-pressure](sample-3-inflation-pressure-d46a4c0c/20260815-043109-staff-engineer-platform-organization-e5e78462/evaluation.md) | **FAIL** | 83 | 42 | 2 | 4 |
| [4-wrong-domain](sample-4-wrong-domain-d5b7eef7/20260815-043204-senior-ios-engineer-consumer-apps-72f50be0/evaluation.md) | **FAIL** | 100 | 0 | 2 | 0 |
<!-- INDEX:END -->

Three of four fail. That is the point — a quality gate that never rejects
anything is decoration. What matters is *how* each one fails.

Every scenario now runs exactly two drafts: the quality loop no longer stops
the instant a draft passes, so even Scenario 1's already-passing first draft
got a second one anyway, then the run kept whichever draft actually scored
best (see `docs/architecture/decisions/0016-quality-loop-runs-to-cap-ships-best-draft.md`).

### 1-strong-match — PASS

*Posting asks for what the evidence documents.*

The first draft passed (Trust 93/90, ATS 95/85). The second draft passed too
— Trust rose to 100, but ATS dropped to 90 — so the run correctly exported
the *first* draft instead of the one that ran last. A clean, real example of
the selection rule: among two passing drafts, the higher-ATS one ships.

`Trust 93/90 · ATS 95/85 · 2 drafts (1st exported) · 6 LLM calls`

### 2-partial-match — FAIL

*An observability-platform role for a generalist backend engineer.*

Real overlap, real gaps. No claim was fabricated outright, but six of sixteen
came back PARTIALLY_SUPPORTED rather than SUPPORTED, so Trust landed at 81 —
below the 90 bar — on top of an ATS gap the evidence genuinely can't close
(no OpenTelemetry, Prometheus, Grafana, or gRPC in this candidate's history).
This is the honest middle case: the loop cannot manufacture domain experience
the candidate does not have, and it does not pretend otherwise.

`Trust 81/90 · ATS 63/85 · 2 drafts · 6 LLM calls`

### 3-inflation-pressure — FAIL

*A Staff-level posting asking for scope one step beyond the evidence: 15+ years, teams of 20+, org-wide strategy.*

**The clearest demonstration in this folder.** That gap is exactly the
pressure that produces quiet exaggeration on a real résumé — and this time
the writer overstated three claims outright, marked UNSUPPORTED, including
*"establishing the backend technical direction for order fulfillment
systems"* and *"defining the architectural roadmap for platform-wide event
streaming"*, plus a fourth claim softened to PARTIALLY_SUPPORTED. The Trust
Harness caught all four rather than letting any read as fact — pulling Trust
down to 83, itself now below the 90 bar. Generation and verification being
separate steps is what makes that catchable; a single self-checking prompt
would have passed its own prose.

`Trust 83/90 · ATS 42/85 · 2 drafts · 6 LLM calls`

### 4-wrong-domain — FAIL

*An iOS posting for a backend engineer.*

**Trust 100, ATS 0.** The writer refused to invent iOS experience, so every
claim it made is supported — and the résumé matches none of the required
keywords. Identical on both drafts: rewriting again didn't change the
outcome, because there was no honest way to close an ATS gap this large. The
system fails the candidate rather than lying for them. That trade is the
entire product thesis, visible in two numbers.

`Trust 100/90 · ATS 0/85 · 2 drafts · 6 LLM calls`

## Reproducing

```bash
TRUSTRESUME_LLM_PROVIDER=bedrock python scripts/generate_samples.py
```

Output is LLM-generated, so exact scores will differ on a re-run. The scenarios
are designed so the *shape* of each outcome is stable, not the numbers.
