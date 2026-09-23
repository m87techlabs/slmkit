# Runbooks

Hands-on companions to each milestone: **what was built, how to check it yourself, and why it
was done that way.** Where `../concepts/` explains ideas, a runbook is something you run from
top to bottom in a terminal, comparing your output against the expected output on the page.

| Doc | Answers | Analogy |
|---|---|---|
| `../concepts/*.md` | Why does this work the way it does? | design doc |
| `../MODEL.md` | What exactly is the model we build? | spec sheet |
| `../STACK.md` | What is each tool, and why is it here? | bill of materials |
| `../decisions/*.md` | What did we choose, and what did we reject? | ADR |
| `runbooks/*.md` | Is it working on *my* machine, right now? | runbook / smoke test |

Each runbook is updated in the same change as the milestone it covers. When a verification step
changes, the runbook changes with it.

Expected output in these pages comes from the **reference machine** (RTX 5080, 16 GB, WSL2; see
`../DESIGN.md` §1). Your numbers will differ on other hardware, and the pages say which numbers
must match exactly and which are only ballpark figures.

| Runbook | Milestone | Status |
|---|---|---|
| [`m0-environment.md`](m0-environment.md) | M0 | ☑ |
| [`m1-engine.md`](m1-engine.md) | M1 | ◐ Phase A |
