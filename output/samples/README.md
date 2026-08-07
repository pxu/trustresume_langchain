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

Regenerated from the committed `evaluation.json` files, never hand-edited —
see `write_index()` in `scripts/generate_samples.py`. Regenerate with:

```bash
python scripts/generate_samples.py --reindex
```

<!-- INDEX:START -->
| Scenario | Result | Trust /90 | ATS /85 | Drafts | Claims flagged |
|---|---|---|---|---|---|
| [1-strong-match](sample-1-strong-match-cc2ac524/20260807-225248-senior-backend-engineer-platform-434bf056/evaluation.md) | **PASS** | 94 | 90 | 1 | 2 |
| [2-partial-match](sample-2-partial-match-0ffef33d/20260807-225449-backend-engineer-observability-platform-fff78ef4/evaluation.md) | **FAIL** | 94 | 53 | 4 | 2 |
| [3-inflation-pressure](sample-3-inflation-pressure-d46a4c0c/20260807-225650-staff-engineer-platform-organization-b55c0d4d/evaluation.md) | **FAIL** | 92 | 50 | 4 | 3 |
| [4-wrong-domain](sample-4-wrong-domain-d5b7eef7/20260807-225826-senior-ios-engineer-consumer-apps-319b2835/evaluation.md) | **FAIL** | 100 | 0 | 4 | 0 |
<!-- INDEX:END -->

Three of four fail. That is the point — a quality gate that never rejects
anything is decoration. What matters is *how* each one fails.

### 1-strong-match — PASS

*Posting asks for what the evidence documents.*

The happy path: the first draft already cleared both thresholds. Two claims
(mentoring, on-call rebuild) came back PARTIALLY_SUPPORTED rather than a clean
sweep — a PASS can still carry flagged claims, since Trust is a mean over
claim statuses, not a unanimous vote.

`Trust 94/90 · ATS 90/85 · 1 draft · 4 LLM calls · $0.32`

### 2-partial-match — FAIL

*An observability-platform role for a generalist backend engineer.*

Real overlap, real gaps. Trust stayed high (94) across all four drafts, but
ATS still fell well short — the evidence has no observability-vendor
specifics (OpenTelemetry, Prometheus, Grafana, gRPC, SLOs), and the writer
never invented them. This is the honest middle case: the loop cannot
manufacture domain experience the candidate does not have, and it does not
pretend otherwise.

`Trust 94/90 · ATS 53/85 · 4 drafts · 10 LLM calls · $1.07`

### 3-inflation-pressure — FAIL

*A Staff-level posting asking for scope one step beyond the evidence: 15+ years, teams of 20+, org-wide strategy.*

**The clearest demonstration in this folder.** That gap is exactly the
pressure that produces quiet exaggeration on a real résumé — and the writer
did stretch, producing plausible filler like *"proven ability to set
technical direction for backend services"* and *"established technical
direction for event-driven backend services and cross-team data
infrastructure."* The Trust Harness flagged all three PARTIALLY_SUPPORTED
rather than letting them pass as fact. Generation and verification being
separate steps is what makes that catchable; a single self-checking prompt
would have passed its own prose.

`Trust 92/90 · ATS 50/85 · 4 drafts · 10 LLM calls · $1.06`

### 4-wrong-domain — FAIL

*An iOS posting for a backend engineer.*

**Trust 100, ATS 0.** The writer refused to invent iOS experience, so every claim it made is supported — and the résumé matches none of the required keywords. The system fails the candidate rather than lying for them. That trade is the entire product thesis, visible in two numbers.

`Trust 100/90 · ATS 0/85 · 4 drafts · 10 LLM calls · $0.87`

## Reproducing

```bash
TRUSTRESUME_LLM_PROVIDER=bedrock python scripts/generate_samples.py
```

Output is LLM-generated, so exact scores will differ on a re-run. The scenarios
are designed so the *shape* of each outcome is stable, not the numbers.
