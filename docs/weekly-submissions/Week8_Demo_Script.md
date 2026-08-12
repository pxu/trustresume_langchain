# TrustResume Demo Video Script

For Week 8: Project Demonstration. The app is already running and verified end to end
(see "Setup" below to restart it if needed). This script assumes about 4 to 6 minutes
of screen recording; trim narration as needed.

## Setup (already done, restart if your machine reboots)

```bash
cd /Users/joe.xu/repo/trustresume_langchain
source .venv/bin/activate

# Terminal 1: backend, real Bedrock provider
uvicorn trustresume.api.server:build_served_app --factory --port 8000

# Terminal 2: frontend
TRUSTRESUME_API_URL=http://localhost:8000 streamlit run src/trustresume/ui/streamlit_app.py
```

Open http://localhost:8501 in your browser. Leave the "User id" field blank so it
resolves to the demo user, whose account already has one uploaded resume
(`AI_Engineer_Resume.docx`) and one job ("Applied AI Engineer at Didero") with two
completed generations, in case you want to show history without waiting on a new run.

A real fix worth knowing before you record: the frontend used to time out after 120
seconds, and a real 4 draft generation against Bedrock takes about 3 minutes, so the
UI used to show a false "could not reach the backend" error even though the run had
actually succeeded and saved. That is fixed now (timeout raised to 300 seconds in
`src/trustresume/ui/api_client.py`), verified by rerunning the exact same flow. If you
see that error anyway, it means something is genuinely wrong and worth mentioning on
camera rather than restarting and hoping it goes away.

## Recording

On macOS: `Cmd+Shift+5` opens the screen recording picker (record a window or the
whole screen, with audio). QuickTime Player's File menu has the same option. Aim your
browser window at a reasonable size before you start; resize it now rather than mid
recording.

## Storyboard

**0:00 to 0:30, intro.** Say what TrustResume is in one or two sentences: it generates
a resume grounded in a candidate's own documents, then an independent agent checks
every claim against that evidence before showing a score to anyone. Mention this is
the live application, not slides.

**0:30 to 1:15, Documents tab.** Show the already uploaded résumé, or upload a new one
live: click "Choose File", pick a résumé from `data/sample_documents/`, set the
document type to RESUME, click Upload. Point out the "Ingested ... as RESUME" message
and the chunk count.

**1:15 to 2:00, Jobs tab.** Either open the existing "Applied AI Engineer at Didero"
job from the dropdown, or paste a new posting into "Job posting" and click "Create
job". Mention that a job is a persisted entity: it runs the Job Description agent once
and reuses that extraction for every generation against it, rather than re-extracting
from scratch each time.

**2:00 to 2:15, trigger a generation.** Click "Generate for this job". While it spins,
narrate the pipeline rather than sitting in silence: evidence retrieval (hybrid vector
and keyword search over the candidate's documents), the Resume Writer drafting from
that evidence, the Trust Harness independently checking every claim, and ATS keyword
scoring, looping up to three rewrites if either score falls short. This step takes
roughly two to three minutes against a real model; if you would rather not show dead
air, skip ahead to the next beat using an already completed run instead of waiting live.

**2:15 to 3:30, the result.** Whichever run you show (fresh or the two already saved),
walk through what's on screen:

- Trust score and ATS score as two metric cards, plus how many iterations it took.
- The real numbers from the run already on file: Trust 92, ATS 38, 3 iterations, 8 LLM
  calls, about 38,000 tokens, 171 seconds, $1.52. A second run on the same job scored
  Trust 86, ATS 50.
- The warning banner: "Hit the rewrite cap without passing, showing the last draft
  anyway." Say plainly that this run did not pass, and that it is shown anyway rather
  than hidden or silently upgraded.
- Scroll to "Missing ATS keywords": point out real gaps like "agentic systems",
  "supply chain", "Django", "Pydantic", words this candidate's actual background does
  not contain. This is the point to make explicit: the writer did not invent
  supply-chain or Django experience to close that gap. It told the truth about what
  the evidence supports and let the ATS score reflect the real mismatch.
- If a claim gets flagged, show the "Unsupported claims" section too; this run had
  zero, which is also worth saying, since a passing Trust score without a flagged
  claim only means the harness agreed the writer stayed grounded, not that it never
  checked.

**3:30 to 4:00, artifact.** Click "Download PDF" or "Download Markdown" on a past
resume and open the file, showing that a real exportable document comes out the other
end, not just numbers on a screen.

**4:00 to 4:30, close.** One sentence on the thesis: separating the agent that writes
from the agent that checks is what makes this trustworthy rather than merely
plausible, and a real run just demonstrated the system choosing an honest, capped
result over a flattering, fabricated one.

## Submitting

Export or trim the recording to an mp4 (QuickTime: File > Export As). If the file is
large, upload it to an unlisted YouTube video, Loom, or a GitHub release asset on
https://github.com/pxu/trustresume_langchain and submit that link instead of the raw
file, per the assignment's "mp4 or link to video location" option.
