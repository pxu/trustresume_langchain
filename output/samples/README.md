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
| [1-strong-match](sample-1-strong-match-cc2ac524/20260807-182047-senior-backend-engineer-platform-1c30830f/evaluation.md) | **PASS** | 91 | 95 | 2 | 3 |
| [2-partial-match](sample-2-partial-match-0ffef33d/20260807-182256-backend-engineer-observability-platform-d6b9277a/evaluation.md) | **FAIL** | 84 | 45 | 4 | 6 |
| [3-inflation-pressure](sample-3-inflation-pressure-d46a4c0c/20260807-182508-staff-engineer-platform-organization-ff4edd86/evaluation.md) | **FAIL** | 85 | 58 | 4 | 4 |
| [4-wrong-domain](sample-4-wrong-domain-d5b7eef7/20260807-182643-senior-ios-engineer-consumer-apps-13ad6e37/evaluation.md) | **FAIL** | 100 | 0 | 4 | 0 |

Three of four fail. That is the point — a quality gate that never rejects
anything is decoration. What matters is *how* each one fails.

### 1-strong-match — PASS

*Posting asks for what the evidence documents.*

The happy path. Note it still took one rewrite to get there — the first draft missed keywords, the loop fed back specifics, and the second draft cleared both thresholds.

`Trust 91/90 · ATS 95/85 · 2 drafts · 6 LLM calls`

### 2-partial-match — FAIL

*An observability-platform role for a generalist backend engineer.*

Real overlap, real gaps. It used every rewrite and still failed on both axes. This is the honest middle case: the loop cannot invent domain experience the candidate does not have, and it does not pretend otherwise.

`Trust 84/90 · ATS 45/85 · 4 drafts · 10 LLM calls`

### 3-inflation-pressure — FAIL

*A Staff-level posting asking for scope one step beyond the evidence: 15+ years, teams of 20+, org-wide strategy.*

**The clearest demonstration in this folder.** That gap is exactly the pressure that produces quiet exaggeration on a real résumé — and the writer did stretch, producing plausible filler like *"Strengthening systems architecture across the organization"*. The Trust Harness flagged it UNSUPPORTED. Generation and verification being separate steps is what makes that catchable; a single self-checking prompt would have passed its own prose.

`Trust 85/90 · ATS 58/85 · 4 drafts · 10 LLM calls`

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
