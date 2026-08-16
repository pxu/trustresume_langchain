# TrustResume: Week 8 Demonstration Speaker Script

Narration for the slides that bookend the live demo, in slide order. The demo walkthrough itself is storyboarded in `Week8_Demo_Script.md`; this covers the framing slides before and after it. The same text is embedded as speaker notes in `Week8_Demonstration.pptx`.

## Slide 1. Title

Hi, I'm Peng Fei Xu. This is the Week 8 demonstration of TrustResume. Everything you're about to see is the live application running against a real production model — not slides, not a mock-up. In one sentence: TrustResume generates a resume grounded in the candidate's own documents, and then an independent agent checks every claim against that evidence before any score is shown to anyone.

## Slide The Idea in One Sentence

Here's the whole idea before we open the app. Generative AI writing a convincing resume is a solved problem — the hard part is trust. Nothing stops a model from quietly stretching the truth to make a draft look better. So TrustResume splits generation from verification into two separate agents. The writer drafts from the candidate's real documents; an independent Trust Harness then checks every single claim against that evidence. The writer never scores itself. The bet the whole project rests on is that the system should rather fail a candidate than lie for them — and I'll show that happening on a real run in a moment.

## Slide What You'll See in This Demo

Quick roadmap for the next few minutes. Four tabs. First, Documents: I'll upload a résumé and you'll see it ingested — parsed, chunked, embedded — into an evidence store scoped to just this user. Second, Jobs: a job here is a saved entity; the posting is extracted once and reused, so we're not re-parsing it every run. Third, Generate: this kicks off the pipeline — hybrid vector-plus-keyword retrieval, the Writer drafting from evidence, the Trust Harness checking each claim, and ATS keyword scoring. It always rewrites at least once more and keeps whichever draft actually scored best, not the first that passed. Finally the result: the real scores, the gaps, any flagged claims, and an actual exportable document.

## Slide Under the Hood: Six Agents, One Quality Loop

Before we dive in, one architecture slide so the demo makes sense. The Streamlit UI talks over HTTP to a FastAPI backend, which drives a LangGraph orchestrator. The orchestrator sequences the agents: Job Description and Evidence Retrieval run once, then a loop of Resume Writer, Trust Harness, and ATS Evaluation, with a cached Candidate Profile feeding the writer. Retrieval is hybrid — part vector similarity for paraphrases, part keyword search for exact product names — over a Chroma vector store plus SQLite full-text search, everything filtered to one user's own data. That's the isolation boundary the trust story depends on.

## Slide LIVE DEMO

Okay — switching to the live app now. This is the real system against a real Bedrock model. I'll walk through Documents, Jobs, and a generation, and then we'll look at what came out.

## Slide The Moment to Watch For

As I run this, here's the one moment I want you to watch for. When a candidate's evidence doesn't actually match the job, a naive generator would just invent the missing experience to score higher. This system won't. You'll see it produce a high Trust score — because every claim it made is genuinely supported — right next to a low ATS score, because the résumé honestly doesn't match the posting's keywords. Trust high, ATS low, on the very same résumé. That's not a failure of the system; that is the system working exactly as designed. And when a run doesn't clear the quality gate, we show it anyway, labeled plainly — never hidden, never quietly upgraded.

## Slide 7. Close

To close: separating the agent that writes from the agent that checks is what makes this trustworthy rather than merely plausible — and you just watched a real run choose an honest, capped result over a flattering, fabricated one. That's the whole point. Thank you — happy to take any questions.
