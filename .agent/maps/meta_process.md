# Meta Process & Prompt Docs Map

> **When to open:** Updating execution workflow docs, contracts, repo routing, agent guidance, or doc-policy checker scripts.

## Primary Files

- `dev/process-tasks.md` — canonical execution mechanics (TDD, quality gates, completion, failure handling)
- `dev/spec-driven.md` — spec-driven prompt template (behavioral guardrails + references)
- `dev/contracts/README.md` + `dev/contracts/*.md` — active implementation contracts
- `dev/archive/` — historical task lists, summaries, and release/safety records
- `REPO_MAP.md` — task routing table
- `.agent/maps/*.md` — minimal-context sub-maps
- `AGENTS.md` — repository guidance for Codex-style agents
- `CLAUDE.md` — repository guidance for Claude Code
- `scripts/check_*.py` — lightweight acceptance/doc-policy/production-safety validators

## Ownership Model

- Keep detailed mechanics in `dev/process-tasks.md`
- Keep prompt behavior constraints in `dev/spec-driven.md`
- Keep active task scope and invariants in `dev/contracts/*.md`
- Keep historical execution artifacts and past release evidence in `dev/archive/`
- Keep routing logic in `REPO_MAP.md`
- Keep sub-map summaries in `.agent/maps/*.md`
- Refresh `REPO_MAP.md` and any affected `.agent/maps/*.md` whenever development changes make routing targets or summaries stale
- Keep agent-entry guidance in `AGENTS.md` and `CLAUDE.md`
- Keep lightweight validation logic in `scripts/check_*.py`

Avoid duplicating the same procedural rules across multiple documents.

## Required Consistency Checks

When these docs change, verify:

1. Canonical source references still point to the same files
2. Quality gate commands are consistent with current project standards
3. Public-contract testing policy references are still accurate
4. `REPO_MAP.md` router entries still point to valid sub-maps
5. Active contract docs in `dev/contracts/` match the current execution workflow
6. Archived task/safety records remain clearly separated under `dev/archive/`
7. Completion-summary expectations in process docs still match agent guidance
8. Acceptance and production-safety check commands are present and referenced

## Optional Verification Command

```bash
python3 scripts/check_doc_policy_consistency.py
```
