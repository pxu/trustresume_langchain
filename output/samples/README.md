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

| Scenario | Result | Trust /90 | ATS /85 | Drafts | Claims flagged |
|---|---|---|---|---|---|
| [1-strong-match](sample-1-strong-match-cc2ac524/20260807-195015-senior-backend-engineer-platform-7d5246e6/evaluation.md) | **PASS** | 91 | 95 | 1 | 3 |
| [2-partial-match](sample-2-partial-match-0ffef33d/20260807-195219-backend-engineer-observability-platform-eb727492/evaluation.md) | **FAIL** | 100 | 42 | 4 | 0 |
| [3-inflation-pressure](sample-3-inflation-pressure-d46a4c0c/20260807-195423-staff-engineer-platform-organization-a2ba1b78/evaluation.md) | **FAIL** | 97 | 17 | 4 | 1 |
| [4-wrong-domain](sample-4-wrong-domain-d5b7eef7/20260807-195558-senior-ios-engineer-consumer-apps-c16d5cc0/evaluation.md) | **FAIL** | 100 | 0 | 4 | 0 |

Three of four fail. That is the point — a quality gate that never rejects
anything is decoration. What matters is *how* each one fails.

### 1-strong-match — PASS

*Posting asks for what the evidence documents.*

The happy path: the first draft already cleared both thresholds.

`Trust 91/90 · ATS 95/85 · 1 draft · 4 LLM calls`

### 2-partial-match — FAIL

*An observability-platform role for a generalist backend engineer.*

Real overlap, real gaps. Every claim stayed honest (Trust 100) across all four drafts, but ATS still fell well short — the evidence has no observability-vendor specifics (OpenTelemetry, Prometheus, Grafana, gRPC), and the writer never invented them. This is the honest middle case: the loop cannot manufacture domain experience the candidate does not have, and it does not pretend otherwise.

`Trust 100/90 · ATS 42/85 · 4 drafts · 10 LLM calls`

### 3-inflation-pressure — FAIL

*A Staff-level posting asking for scope one step beyond the evidence: 15+ years, teams of 20+, org-wide strategy.*

**The clearest demonstration in this folder.** That gap is exactly the pressure that produces quiet exaggeration on a real résumé — and the writer did stretch, producing plausible filler like *"establishing shared systems architecture standards"*. The Trust Harness flagged it PARTIALLY_SUPPORTED rather than letting it pass as fact. Generation and verification being separate steps is what makes that catchable; a single self-checking prompt would have passed its own prose.

`Trust 97/90 · ATS 17/85 · 4 drafts · 10 LLM calls`

### 4-wrong-domain — FAIL

*An iOS posting for a backend engineer.*

**Trust 100, ATS 0.** The writer refused to invent iOS experience, so every claim it made is supported — and the résumé matches none of the required keywords. The system fails the candidate rather than lying for them. That trade is the entire product thesis, visible in two numbers.

`Trust 100/90 · ATS 0/85 · 4 drafts · 10 LLM calls`

## Reproducing

```bash
TRUSTRESUME_LLM_PROVIDER=bedrock python scripts/generate_samples.py
```

Output is LLM-generated, so exact scores will differ on a re-run. The scenarios
are designed so the *shape* of each outcome is stable, not the numbers.
