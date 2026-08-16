# TrustResume — Final Presentation Script

**Evidence-Based Resume Generation Using RAG, Multi-Agent AI, and Trust Verification**
Peng Fei Xu · MSAI 699 Capstone · University of the Cumberlands

**Target: 8–9 minutes · 10 slides.** Read at a calm pace (~140 words/min).
Text in *(italics)* is a delivery cue, not something to read aloud.

---

## Slide 1 — Title & Hook  ·  ~0:30

*(Pause. Make eye contact before you speak. Let the question land.)*

> Let me start with a question. If an AI tool writes your resume in seconds —
> **who checks that every claim in it is actually true?**
>
> Right now, nobody does. That gap is what my project closes.
>
> I'm Peng Fei Xu, and this is **TrustResume**. It generates a resume from real
> evidence, and then **independently verifies every claim before delivering it**.
> I'll cover the problem, how it works, the results, and what it means for the
> people who use it.

---

## Slide 2 — The Problem & Why It Matters  ·  ~1:00

> Generative AI can write a convincing resume in seconds — that part is solved.
> The real problem is **trust**. Nothing stops the model from stretching the
> truth, and a fabricated claim looks exactly like a real one.
>
> Why does this matter? *(gesture to the three cards)*
> - For **candidates**, one claim they can't back up can cost them their credibility.
> - For **employers**, it means wasted time and bad hires.
> - For **organizations**, unverifiable AI content is a real legal and reputational risk.
>
> So this isn't just a technical problem — it affects everyone in hiring.

---

## Slide 3 — The Innovation  ·  ~1:15

*(Slow down — this is the key slide. Point to the two boxes.)*

> Here is the core innovation — and it is **not** the tech stack. It's this one
> idea: **every claim is independently verified against the candidate's own
> evidence, before the resume is delivered.**
>
> Most AI tools let the model write, and then effectively trust itself. We split
> those two jobs. A **Writer** agent generates the resume. Then a completely
> **separate Verifier** checks every single claim against the evidence.
>
> **The writer never grades its own homework.** That independent verification
> layer is the whole point of this project.

---

## Slide 4 — How It Works (Methodology)  ·  ~1:00

*(Trace the loop on the diagram as you talk.)*

> Here's the method in one picture. Two steps run once: we read the job, and we
> retrieve the candidate's evidence. Then a **loop** begins.
>
> The Writer drafts. The Verifier — our Trust Harness — scores every claim as
> supported, partly supported, or unsupported. An ATS check scores keyword
> coverage against the job. The system writes feedback from whatever is weakest,
> tries again, and in the end **keeps the best-scoring draft, not just the last one.**
>
> Generation and verification are always separate steps in that loop.

---

## Slide 5 — System Architecture  ·  ~1:00

*(Walk left-to-right, then top-to-bottom.)*

> A quick architecture walkthrough. On the left is the path a request takes: the
> web UI talks over HTTP to a thin API layer, which calls one application
> service, which calls the Orchestrator.
>
> In the middle are the **six agents** the Orchestrator runs in order — the green
> ones use **no AI model at all**, so retrieval and scoring are plain, testable
> code, not a black box. At the bottom is storage, with hybrid search that
> combines meaning-based and keyword search.
>
> The key point for you: every search is scoped to **one user's own data**, and
> the scoring logic has no AI model inside it — so the trust score is
> **deterministic and auditable**.

---

## Slide 6 — Results (KPIs)  ·  ~1:00

*(Let each big number breathe.)*

> Now the results — and the point is that we **measured**, we didn't assume.
> Three headline numbers:
> - Retrieval finds the right evidence **100%** of the time on our labeled test set.
> - The verifier is **83.3%** accurate.
> - And we have **99.2%** test coverage — 500 automated tests on every change.
>
> Here's the honest story behind that 83.3%: the first version scored only 50%,
> because we just told it to "be strict" without defining what that meant. We
> caught that with our own evaluation harness and calibrated it up. And the
> **dangerous error — a false claim marked true — stayed at zero the whole time.**

---

## Slide 7 — The Proof  ·  ~1:15

*(This is your money slide. Point to 100, then to 0.)*

> This is the slide I most want you to remember. We took a **backend engineer's**
> real resume and applied it to an **iOS Developer** job — a candidate with
> **zero** iOS experience.
>
> Look at the two numbers. The **Trust Score is 100** — every claim the system
> made was true. The **ATS match is 0** — there was no matching experience, and
> the writer **refused to invent any**.
>
> So instead of faking an iOS background to pass, the system reported an honest
> failure. **It would rather fail you than lie for you.** That's the entire
> thesis of this project, proven on a real production model.

---

## Slide 8 — Responsible AI & What's Next  ·  ~0:50

> Two quick things, because this touches real careers.
>
> First, **responsible design**: every outside input is treated as data, never
> as an instruction — that defends against prompt injection; every user's data
> is isolated; and the same verification that catches fabrication also keeps the
> system **fair**.
>
> Second, **what's next**, in four areas: real authentication and security;
> better retrieval quality; higher trust accuracy from a larger dataset; and
> scalability plus more languages. I'll be honest — real login isn't built yet,
> and it's the top priority.

---

## Slide 9 — Conclusion  ·  ~0:45

*(Slow, confident close. This is the last thing they'll remember.)*

> To conclude. The big idea is simple: **AI-generated content can be useful
> without sacrificing trust — and the key is independent verification.**
>
> We didn't just claim that; we **proved it** on a real model that chose an
> honest score over a fabricated resume. For candidates, that means a resume they
> can stand behind. For employers, claims they can actually trust.
>
> And this pattern — grounded generation plus independent verification — reaches
> far beyond resumes, to any AI-generated content where the truth matters.
> Thank you.

---

## Slide 10 — Thank You / Questions  ·  ~0:15

> Thank you very much for your time. I'd be glad to take any questions.

---

### Timing summary

| # | Slide | Time |
|---|-------|------|
| 1 | Title & Hook | 0:30 |
| 2 | Problem & Why It Matters | 1:00 |
| 3 | The Innovation | 1:15 |
| 4 | How It Works | 1:00 |
| 5 | System Architecture | 1:00 |
| 6 | Results (KPIs) | 1:00 |
| 7 | The Proof | 1:15 |
| 8 | Responsible AI & What's Next | 0:50 |
| 9 | Conclusion | 0:45 |
| 10 | Thank You | 0:15 |
| | **Total** | **~8:30** |

### Q&A prep — likely questions
- **"How is 'trust' actually scored?"** Every claim is labeled supported / partly / unsupported against retrieved evidence; the score is code-computed, not the model's opinion, with temperature pinned to zero so it's repeatable.
- **"Couldn't the verifier be wrong too?"** Yes — that's why we measure it against hand-labeled ground truth (83.3%), and why the false-positive rate (false marked true) is the metric we drove to zero.
- **"Why not just prompt one model to 'be honest'?"** A model grading itself has no independent check. Separating writer and verifier is what makes claims *checkable* instead of just *plausible*.
- **"What does it cost / how fast?"** ~$0.51–$0.60 per resume, measured per run (tokens, latency, cost tracked end-to-end).
