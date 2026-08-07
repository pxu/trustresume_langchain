# ADR-0014: Resolve the user per request from a header

## Status
Accepted. Supersedes the hardcoded single-user demo id at the API layer; the
storage-level isolation it exercises is unchanged (ADR-0001).

## Context
Per-user isolation is the storage layer's central invariant. Every retrieval
carries a `user_id` metadata filter (ADR-0001 calls it "the isolation
boundary"), every repository method is user-scoped, ingestion dedup is
deliberately scoped `(user_id, content_hash)` so two users uploading the same
résumé template don't collide, and every facade method that takes a `job_id` /
`resume_id` / `document_id` verifies ownership before acting.

None of it could be demonstrated. `api/server.py` hardcoded `DEMO_USER_ID` on
all 21 routes, so **every request was the same user** — no test and no demo
could show that two users are actually isolated. The project's most-emphasized
design property was, at the HTTP boundary, unfalsifiable.

That's a testability gap, not an authentication gap. A capstone doesn't need a
login system; it does need its headline claim to be checkable.

## Decision
Resolve the caller from an `X-User-Id` header via one FastAPI dependency
(`resolve_user`), injected into every user-scoped route as `CurrentUser`.

- **Absent or blank header → `DEMO_USER_ID`**, so the walk-up-and-try-it flow
  and every existing client keep working unchanged.
- **Unknown ids are created on first use**, so a client picks its own id with
  no signup step.
- **Ids are validated** (`[A-Za-z0-9._-]{1,64}`) rather than trusted blindly.
  Queries are parameterized, so this isn't an injection defense — it's about
  not letting a stray or hostile header mint database keys that are
  megabyte-long or full of control characters.

This is **identity, not authentication**: the header is trusted as sent.
Making it real auth means replacing this one function with a token-verifying
dependency — every store call underneath is already user-scoped, which is the
point.

The Streamlit UI gets a sidebar "User id" field, and `TrustResumeClient` sets
the header once on its `requests.Session` (never per method — a client that
forgot it on one call would silently read the demo user's data).

## Consequences
- Isolation is now provable, and proven: `test_api.py` asserts that two users
  can't see each other's documents, that search doesn't leak evidence across
  users, and that one user's resume id 404s for another. Those tests could not
  have been written before.
- Switching the user id in the UI switches to a completely separate workspace,
  which makes ADR-0001 demonstrable in a live demo rather than only in prose.
- `resolve_user` must live at **module scope** and reach the facade through
  `request.app.state`, not close over `create_app`'s `app_facade`. With
  `from __future__ import annotations` every annotation is a string, so a
  `CurrentUser` alias defined inside `create_app` is unresolvable from module
  globals — FastAPI silently degrades that to "unknown query parameter" and
  *every route returns 422*. This was caught by running the app, not by mypy
  or ruff, and the docstring says so to keep it from being "cleaned up" back
  into a closure.

## Alternatives considered
- **Real authentication (JWT/OAuth).** Out of scope for a personal capstone,
  and it would put a token issuer between a reviewer and a working demo. The
  dependency-shaped seam means it's a one-function change later.
- **A `user_id` query parameter or body field.** Rejected: it would have to be
  added to every request schema and every UI call site, and it invites clients
  to treat identity as per-call data rather than per-session context.
- **Leaving the demo user in place.** Rejected — that's the status quo whose
  only real cost is that the project's own central claim stays untested.
