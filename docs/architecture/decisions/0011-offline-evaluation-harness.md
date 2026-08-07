# ADR-0011: An offline evaluation harness with labeled ground truth

## Status
Accepted. New — no equivalent decision in the original repo, which had no
offline evaluation of any kind.

## Context
This project had no way to answer "did that change make the system better?"

Two things it *did* have look like evaluation but aren't:

- **The quality gate** (`Trust >= 90 AND ATS >= 85`, ADR-0005) scores one
  generated résumé for one user, at runtime, as product output. It says
  nothing about whether the system improved.
- **`evaluation/scorer.py`** computes ATS keyword coverage — again, a number
  shown to the user about their draft.

That left two components able to regress silently:

1. **Retrieval.** A regression here doesn't announce itself. The writer simply
   grounds fewer claims and the draft still reads fine, so the visible Trust/ATS
   scores may barely move. `docs/code-walkthrough.md` §7.3 already conceded
   this in writing: the switch to `RecursiveCharacterTextSplitter` changed
   chunk boundaries, "so retrieval hits and scores could change in theory —
   a production system should measure before/after." Nothing could.
2. **The Trust Harness.** This is worse, and is the project's central claim.
   The runtime Trust score is *computed from whatever the harness reported*
   (`TrustReport.compute_score` averages the harness's own verdicts). A harness
   that classifies every claim SUPPORTED yields a perfect Trust score and a
   completely undetectable failure — the metric cannot see its own blind spot.

## Decision
Add `src/trustresume/evals/`: two suites scored against committed, labeled
datasets in `evals/datasets/*.jsonl`, run offline via
`python -m trustresume.evals`.

**Retrieval suite** — recall@k, precision@k, MRR, hit rate over a labeled
corpus, ingested through the *real* parse → clean → chunk → embed path (a
harness that evaluated pre-chunked text would hide exactly the regressions
it exists to catch). Chunk hits collapse to document level, since relevance is
labeled per document.

**Trust suite** — each case pins one claim against known evidence with a known
correct verdict, scored as multi-class classification (accuracy, macro-F1,
per-label P/R/F1, confusion matrix).

Three choices inside that are load-bearing:

- **Macro-F1 alongside accuracy.** The label distribution is skewed toward
  SUPPORTED, so a harness that never says UNSUPPORTED scores ~80% accuracy
  while failing at its only job. Macro-F1 weights rare classes equally and
  refuses to be fooled; there's a test asserting exactly that
  (`test_classificationMetrics_alwaysPredictingMajority_exposedByMacroF1`).
- **Too-lenient errors counted separately** (`dangerous_errors`). Calling an
  unsupported claim supported ships a fabrication to a user; flagging a true
  claim costs one rewrite iteration. Reporting both as one error rate would
  erase the asymmetry the whole project is built on.
- **Unanswerable queries excluded from precision/MRR/hit-rate** but kept for
  recall. Those metrics are undefined when there's nothing to find (standard
  IR practice), and counting such a query as a miss would penalize correct
  behavior. Their real signal — *did it return junk anyway* — is reported
  separately as `unanswerable_results`.

Everything except the CLI is dependency-injected, so the harness's own logic
(label mapping, rank collapsing, verdict reduction) is unit-tested offline
against fakes. `src/trustresume/evals/cli.py` is the one module that builds real dependencies,
and is `omit`-ted from the coverage gate for the same reason `poc/` is.

## Consequences
- A retrieval or prompt change now produces a comparable number.
  `evals/baselines/latest.json` records the current one; `evals/README.md`
  explains how to read it.
- The datasets are the harness's weakest point and its main maintenance cost:
  they're synthetic, single-annotator, and small (10 documents, 9 queries, 12
  claims). They're big enough to catch a regression, not big enough to
  certify quality — `evals/README.md` says so rather than implying otherwise.
  Every case carries a `note` naming which capability it probes, so a
  regression points at a cause rather than just a lower number.
- Dataset integrity is enforced by tests (unknown `doc_id`, duplicate ids,
  full label coverage): a typo'd label silently depresses recall forever and
  every later comparison inherits the error.
- The trust suite needs a real provider. Under `TRUSTRESUME_LLM_PROVIDER=test`
  it's skipped with an explanation rather than reporting a misleading 0 (the
  offline fake returns an empty claim list, which scores as UNSUPPORTED
  everywhere).
- The retrieval suite needs no credentials — fastembed runs locally — so the
  metric most likely to regress is also the cheapest to check.

## Alternatives considered
- **An LLM-as-judge suite instead of labeled data.** Cheaper to author, but it
  evaluates a verifier with an unverified verifier, which is the same circular
  problem this ADR exists to break. Worth adding *on top of* ground truth
  later, not instead of it.
- **Reusing the sample résumés in `data/` as the corpus.** Rejected: real
  documents make relevance judgments debatable, and `.docx`/`.pdf` parsing
  would put an unrelated dependency in the measurement loop.
- **Wiring the suites into CI as a gate.** Rejected for now: the retrieval
  suite would fit, but a hard threshold on a 9-query dataset would mostly
  produce false alarms. Run it before and after a retrieval change instead.
