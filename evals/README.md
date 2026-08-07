# evals/ — evaluation datasets, baselines, and results

**Two locations, one system** — worth knowing before you go looking:

| Path | Holds |
|---|---|
| `evals/` (here) | the labeled **datasets**, the recorded **baseline**, and this guide — the things a human reads and edits |
| `src/trustresume/evals/` | the **code**: metrics, dataset loading, both evaluators, and the `python -m trustresume.evals` CLI |

Same split as `data/` vs `src/trustresume/ingestion/`, or `config/` vs
`src/trustresume/api/model_factory.py`: inputs a person curates live at the
repo root; the code that consumes them lives in the package.

Answers the question the runtime quality gate cannot: **is the system getting
better, or just different?**

The quality gate (`Trust >= 90 AND ATS >= 85`) scores one generated résumé for
one user. It cannot tell you whether last week's change to the chunker, the
embedding model, the RRF constant, or a prompt made the *system* better — and
in a RAG pipeline, a quiet retrieval regression doesn't announce itself: the
writer simply grounds fewer claims, and the draft still looks fine.

Two suites, matching the two things that silently regress:

| Suite | What it scores | Why it caps everything downstream |
|---|---|---|
| **Retrieval** | recall@k, precision@k, MRR against a labeled corpus | The writer can only ground claims in evidence retrieval actually surfaced. Recall is the pipeline's ceiling. |
| **Trust Harness** | classification accuracy + macro-F1 against labeled claims | The runtime Trust score is *computed from whatever the harness said*. A harness that rubber-stamps everything produces a perfect score and an undetectable failure. |

## Running it

```bash
# Retrieval only — no credentials needed (fastembed runs locally, ~10s)
python -m trustresume.evals --suite retrieval

# Both suites, against a real provider
TRUSTRESUME_LLM_PROVIDER=bedrock python -m trustresume.evals --suite all

# Record the run to compare a future change against
python -m trustresume.evals --suite all --save evals/baselines/latest.json
```

Each run builds throwaway in-memory SQLite + an ephemeral Chroma collection, so
it never reads or writes your real `trustresume.db` / `chroma_data` — otherwise
the numbers would depend on whatever you happened to have ingested.

The trust suite is **skipped** under `TRUSTRESUME_LLM_PROVIDER=test`: the
offline fake returns an empty claim list, which scores as UNSUPPORTED for every
case. That's correct behavior, not a bug, but it says nothing about a real
model, so the CLI says so instead of printing a misleading 0.

## Current baseline

`evals/baselines/latest.json`.

### Retrieval — k=8, hybrid (vector + BM25, RRF), `BAAI/bge-small-en-v1.5`

```
recall@k     1.000
precision@k  0.156
MRR          0.938
hit rate     1.000
unanswerable 1 query, 8.0 results returned on average
```

`k=8` is not arbitrary: it is `EvidenceRetrievalAgent`'s `DEFAULT_TOP_K`, the
number of chunks a real generation actually retrieves. A retrieval metric
measured at a depth the pipeline never uses describes nothing — the two
constants are wired together so they can't drift apart.

How to read these:

- **recall 1.000** — every labeled document was retrieved within the top 8,
  including `q3` (which never says "Kubernetes", only describes it — vector
  search's job) and `q2` (an exact product name the embedder treats as
  interchangeable with its competitors — keyword search's job). This is the
  concrete evidence that hybrid retrieval (ADR-0010) earns its complexity.
- **precision 0.156** — expected, not a problem: most queries have exactly one
  relevant document, so `k=8` caps precision at 0.125 for them. Precision here
  is useful as a *relative* number across configurations, not as an absolute.
  (It is mechanically lower than the same measurement at `k=5` — which is the
  reason to fix the cut-off to what production uses and leave it there.)
- **MRR 0.938** — the right document is nearly always ranked first.
- **unanswerable: 8.0 results returned** — the one deliberately unanswerable
  query (`q9`) still gets a full 8 results back. Retrieval has no notion of "no
  good answer", so it always fills `k`. That's the honest weak spot this
  dataset exposes; a relevance threshold would be the fix if it ever matters.

Queries with no relevant document are excluded from precision/MRR/hit-rate
(those metrics are undefined when there's nothing to find — standard IR
practice) but still count for recall, where "missed nothing" is genuinely 1.0.

### Trust Harness — Bedrock, `global.anthropic.claude-opus-4-6-v1`, temperature 0

```
accuracy   0.500
macro F1   0.489
                     P     R    F1   n
PARTIALLY_SUPPORTED  0.40  0.40 0.40  5
SUPPORTED            1.00  0.25 0.40  4
UNSUPPORTED          0.50  1.00 0.67  3

too-lenient errors: 0
```

**The harness has a systematic one-notch-strict bias.** Every one of the six
errors is in the same direction, and every one is off by exactly one severity
step:

| Case | Claim | Expected | Predicted |
|---|---|---|---|
| t2, t3, t11 | true statements | SUPPORTED | PARTIALLY_SUPPORTED |
| t4, t6, t7 | inflated statements | PARTIALLY_SUPPORTED | UNSUPPORTED |

The confusion matrix shows the same thing structurally: the `SUPPORTED` row
and the `UNSUPPORTED` column are where everything piles up, and
`confusion[X][more-lenient-than-X]` is **0 everywhere**.

Three things follow, and they're the reason this suite exists:

1. **The direction is the safe one.** Zero fabrications were passed:
   UNSUPPORTED recall is 1.000 — every genuinely unsupported claim was caught.
   A verifier that errs strict costs rewrite iterations; one that errs lenient
   ships a lie. If you only look at accuracy (0.500) you'd conclude the harness
   is broken; the `dangerous_errors` count is what says it's *miscalibrated*,
   not *failing*.
2. **It has a real cost.** t11 is the clearest case: the claim is "Ran 23
   incident postmortems over 18 months" and the evidence literally says "ran 23
   postmortems over 18 months" — a verbatim restatement, judged
   PARTIALLY_SUPPORTED. Under-crediting true claims drags the Trust score down,
   which sends genuinely fine drafts back through the rewrite loop and burns
   LLM calls (and, at the cap, produces an unnecessary rejection).
3. **The fix is a prompt change, not a code change** — tighten what
   `trust_verification/verifier.py` tells the model SUPPORTED means, especially
   that a claim restating the evidence, or generalizing something the evidence
   entails, is fully supported. **Re-run this suite before and after**; that's
   the entire point of having a baseline. Don't change the prompt and the
   baseline in the same commit.

This measurement is not something the runtime Trust score could ever have
surfaced: that score is computed *from* these verdicts, so a systematically
strict harness just looks like "drafts that need more rewrites."

Caveats worth stating plainly: 12 cases is small, the labels are
single-annotator, and at least two (t6 — one false component in a compound
claim; t12 — teaching beginners vs. mentoring juniors) are genuinely debatable.
Treat these numbers as a **regression detector**, not a certification.

## The datasets

`datasets/*.jsonl`, one record per line, `#` comment lines allowed.

- **`retrieval_corpus.jsonl`** — 10 synthetic career-evidence documents,
  written so that relevance is decidable by a human and so specific failure
  modes are probeable (near-duplicates, exact product names, paraphrases with
  zero shared vocabulary, plausible-but-irrelevant distractors).
- **`retrieval_queries.jsonl`** — 9 queries with relevance labels. Phrased the
  way this system actually queries (`retrieval/query.py` builds one string from
  a posting's title + skills + keywords), not as natural-language questions.
  Every case has a `note` saying *which capability it probes*, so a regression
  points at a cause rather than just a lower number.
- **`trust_claims.jsonl`** — 12 labeled claims. Deliberately weighted toward the
  SUPPORTED/PARTIALLY_SUPPORTED boundary, because that's where a résumé actually
  lies (quiet inflation of scope, seniority, and numbers) and where a lenient
  harness does real harm. Flat fabrications are the easy cases and are included
  as a floor, not as the point.

### Adding cases

Append a line. Then run `pytest tests/unit/test_evals.py` — it validates the
committed datasets (unknown `doc_id`s, duplicate ids, full label coverage),
because a typo'd label silently depresses recall forever and every later
comparison inherits the error.

## Why "too-lenient" errors are counted separately

The trust suite reports a `dangerous_errors` count: cases where the harness was
*more permissive* than the label. Those are the errors that ship a fabrication
to a user. The opposite error — flagging a true claim — only costs one rewrite
iteration. Reporting both as one undifferentiated error rate would hide the
asymmetry that this entire project is built around.

Macro-F1 is reported alongside accuracy for the same reason: the label
distribution is skewed toward SUPPORTED, so a harness that never says
UNSUPPORTED can look ~80% accurate while failing at its only job.
