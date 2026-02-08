# Meta Process & Prompt Docs Map

> **When to open:** Updating execution workflow docs, prompt templates, or agent guidance files.

## Primary Files

- `dev/process-tasks.md` — canonical execution mechanics (TDD, quality gates, completion, failure handling)
- `dev/spec-driven.md` — spec-driven prompt template (behavioral guardrails + references)
- `dev/memory.md` — persistent user preferences and mistakes log
- `dev/task-summaries.md` — concise summary log for completed parent tasks
- `dev/production-safety-gate.md` — pre-production safety checklist and evidence
- `dev/skills/*/SKILL.md` — workflow automation skills
- `REPO_MAP.md` — task routing table
- `AGENTS.md` — repository guidance for Codex-style agents
- `CLAUDE.md` — repository guidance for Claude Code

## Ownership Model

- Keep detailed mechanics in `dev/process-tasks.md`
- Keep prompt behavior constraints in `dev/spec-driven.md`
- Keep persistent preferences/mistakes in `dev/memory.md`
- Keep completed-task summaries in `dev/task-summaries.md`
- Keep release evidence in `dev/production-safety-gate.md`
- Keep skill-level automation instructions in `dev/skills/*/SKILL.md`
- Keep routing logic in `REPO_MAP.md`
- Keep agent-entry guidance in `AGENTS.md` and `CLAUDE.md`

Avoid duplicating the same procedural rules across multiple documents.

## Required Consistency Checks

When these docs change, verify:

1. Canonical source references still point to the same files
2. Quality gate commands are consistent with current project standards
3. Public-contract testing policy references are still accurate
4. `REPO_MAP.md` router entry remains valid
5. Memory file (`dev/memory.md`) is referenced by process/prompt docs and remains secret-free
6. Summary log (`dev/task-summaries.md`) is referenced by process/prompt docs and updated per parent completion
7. Parent-task summary is also posted in chat output
8. Acceptance and production-safety check commands are present and referenced

## Optional Verification Command

```bash
python3 scripts/check_doc_policy_consistency.py
```
