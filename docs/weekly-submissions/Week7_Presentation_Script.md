# TrustResume: Final Presentation Speaker Script

Full narration, in slide order. Read-through pace (~140 wpm) targets roughly 7-8 minutes, inside the assignment's 5-10 minute window.
The same text is embedded as speaker notes in Week7_Final_Presentation.pptx. Use whichever is more convenient while recording.

## Slide 1. Title

Good [morning/afternoon] everyone, thank you for the time. I'm Peng Fei Xu, presenting TrustResume, an evidence-based resume generation system built on retrieval-augmented generation, multi-agent AI, and an independent trust-verification layer. This is the final presentation for my MSAI 699 capstone. I'll walk through the problem, how the system works, what the data shows, and where it goes from here.

## Slide The Problem: AI Can Write a Resume. Should You Trust It?

Let's start with the problem. Generative AI can write a convincing resume in seconds. That part is solved. The problem is nothing stops it from quietly stretching the truth: bumping '2 years' to 'senior', turning a real 64 percent improvement into a rounder 90 percent. That's the easiest way for a model to make a draft look better, so without a check, it happens by default. And the cost isn't abstract: candidates lose credibility, employers make bad hiring decisions, and every legitimate use of AI in hiring gets painted with the same brush. So the question driving this project: can we build a system that generates a strong resume and can prove, claim by claim, that it's true?

## Slide The Solution: Generate, Then Independently Verify

The solution is architectural. First, retrieval-augmented generation: instead of inventing a resume, we retrieve the candidate's own documents and ground the draft in that evidence. Second, the key idea: generation and verification are separate agents. The writer never grades its own homework. An independent Trust Harness checks every claim against the evidence, and a resume only ships once it clears a Trust gate and an ATS gate; otherwise the system rewrites, up to three times. The bet this project is built on: it should rather fail a candidate than lie for them. I'll show a real example later.

## Slide How It Works: Six Agents, One Quality Loop

Here's the architecture. Job Description and Evidence Retrieval each run once. Retrieval uses a hybrid search, part vector similarity, part keyword, so it catches both a paraphrase and an exact product name. Those feed a loop: the Resume Writer drafts, the Trust Harness scores every claim as supported, partially supported, or unsupported, and ATS Evaluation scores keyword coverage. If both clear the bar, we're done; if not, the system builds specific feedback from what failed and tries again, up to three rewrites. A sixth agent, a cached Candidate Profile, feeds the writer without being recomputed every time. This all runs as a graph, LangGraph, which makes the loop and the stopping condition explicit and testable, not buried in ad hoc code.

## Slide Built for Reliability, Not Just a Demo

A few engineering choices matter here because they make this deployable, not just a proof of concept. It's provider-agnostic: it runs on AWS Bedrock, OpenAI, or Google Gemini, chosen by configuration, not by rewriting code, so there's no vendor lock-in. Sampling temperature is pinned to zero, because the headline promise is a deterministic, code-computed trust score. If the same resume could score differently on two runs, that promise wouldn't hold. Models are tiered by role: a cheap model handles simple extraction, a stronger one is reserved for writing and for the verifier, the one place skimping actually hurts. And every request is scoped to one user's own data: the isolation boundary the whole trust story depends on.

## Slide How We Know It Works: Measuring, Not Assuming

How do we know this works, rather than just hoping it does? Two layers. First, a 474-test automated suite at 99.4 percent coverage, checking correctness offline on every change. But correctness isn't quality, so there's a second, separate evaluation harness scoring the system against labeled ground truth: does retrieval surface the right evidence, does the Trust Harness classify claims correctly? That second question matters most, because the score a user sees comes straight from whatever the Harness reports. A verifier that marks everything supported would score perfectly and fail invisibly. So we score the verifier itself, against labeled examples, on a real model.

## Slide The Numbers: Retrieval Finds It, Verification Catches It

Here's what the data shows. On the left: retrieval finds the right evidence 100 percent of the time in our labeled test set. The Trust Harness chart is more interesting: the first version of the verifier's instructions only scored 50 percent accuracy, because it said 'be strict' without defining what that meant, so it downgraded almost everything by one notch. Explicit definitions pushed accuracy to over 83 percent, and the dangerous error, something false marked true, stayed at zero the whole time. On the right: four real runs against a production Bedrock model. Three of four fail the quality gate, on purpose. A gate that never says no isn't checking anything.

## Slide The Proof: It Would Rather Fail Than Lie

This is the slide I most want you to remember. Same candidate, four job postings, run against a real production model. Scenario one is a genuine match: it passes cleanly. Scenario four matters most: an iOS posting for a backend candidate with zero iOS experience. The writer refused to invent any. Trust score: a perfect 100, because every claim was true. ATS score: zero, because the resume matches no required keyword. Trust 100, ATS 0, same résumé. That's not a bug. That's the entire product thesis, visible in two numbers. And scenario three shows the harness earning its keep under pressure: asked for scope beyond the evidence, the writer drifted toward exaggeration, and verification caught every one of those claims before anyone saw them.

## Slide Quality Engineering Behind the Scenes

A quick word on the engineering behind this, because it's what makes the last slide's numbers trustworthy rather than lucky. 474 automated tests, over 99 percent coverage, enforced on every change through continuous integration. Every generation is also instrumented: tokens, time, and dollar cost per step, which in practice runs about 32 cents to just over a dollar per résumé. And I'll be candid: several of the most important bugs, like a real model's output crashing the PDF export, or a cost-reporting gap that silently hid Bedrock's real cost, only showed up running the actual system against a real model. No unit test caught them, which is why we kept real end-to-end testing in the loop.

## Slide Responsible by Design

A few words on responsible design, since this touches real people's careers. Security: any external text, such as a posting or an uploaded document, is tagged as data, never an instruction, defending against prompt injection. Fairness: the same verification that catches fabrication also protects against a subtler bias: an unchecked generator would favor whichever candidate's background happens to embellish most convincingly. Privacy: every record and search is scoped to one user's own data. And I want to be upfront: the deployed API currently identifies a caller from a self-reported header. That's enough to prove isolation works, but not real authentication. That's the clearest gap before a real multi-tenant deployment.

## Slide What's Next

So what's next? First, real authentication and rate limiting, needed before this serves multiple real customers safely. Second, a relevance threshold on retrieval, so a question with no good answer returns fewer results instead of forcing a full set. Third, growing the labeled dataset around the two specific claim types the verifier still occasionally misjudges. And fourth, broader language support on export, plus a clear path to scale beyond a single server. None of this is open research. It's concrete, scoped engineering work.

## Slide Key Takeaways

To close, three takeaways. One: generation and verification don't have to be the same step trusting itself. Separating them is what makes an AI system's claims checkable, not just plausible. Two: we proved it, not just claimed it. A real model, faced with a job it couldn't honestly match, chose a perfect trust score over inventing experience, and our own evaluation harness caught and let us fix a real flaw in the verifier before it reached a user. Three: this pattern isn't specific to résumés. Any AI system generating content where truthfulness matters can ground it in real evidence and check it independently before anyone sees it. Thank you, and happy to take questions.

## Slide 13. Thank You

Thank you again for your time. I'd be glad to take any questions.
