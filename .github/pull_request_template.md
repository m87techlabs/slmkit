## What and why

<!-- What this changes, and why. Say what you rejected when the choice wasn't obvious. -->

## Checklist (CONTRIBUTING.md)

- [ ] `make lint` and `make test` pass (CI runs both); `make test-gpu` too if training code changed
- [ ] The engine (`src/slmkit/`) doesn't reference any project (rule 2); a Project API change has an ADR
- [ ] Docs in the same change: the concept page and runbook for what was built, with real output
- [ ] New abbreviations are in `docs/GLOSSARY.md`; new tools, libraries or services in `docs/STACK.md`
- [ ] Every number in docs comes from a run or a command that was actually run
- [ ] Conventional commit subjects; no co-author or generated-by trailers
