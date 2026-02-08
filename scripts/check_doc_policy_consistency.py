#!/usr/bin/env python3
"""Validate key policy anchors across process and prompt docs.

This script is intentionally lightweight and string-based. It helps catch
accidental drift when updating agent/process guidance documents.
"""

from __future__ import annotations

from pathlib import Path
import sys


REQUIRED_SNIPPETS: dict[str, list[str]] = {
    "dev/process-tasks.md": [
        "## Quality Gates (Canonical)",
        "## Acceptance Contract (Required Per Parent Task)",
        "## Production Release Gate (Required Before Production Deployments)",
        "## Memory Hygiene Gate",
        "## Task Metadata Standard (for New/Updated Tasks)",
        "Post the same concise summary in chat",
        "ACCEPTANCE_CHECK_COMMAND",
        "PRODUCTION_SAFETY_CHECK_COMMAND",
        "MEMORY_HYGIENE_COMMAND",
    ],
    "dev/spec-driven.md": [
        "## 0. Bootstrap (Mandatory)",
        "## 3. Execution Protocol",
        "## 4. Testing Boundaries",
        "{{TASK_ID}}",
        "{{MEMORY_PATH}}",
        "{{SUMMARY_LOG_PATH}}",
        "{{ACCEPTANCE_CHECK_COMMAND}}",
        "{{PRODUCTION_SAFETY_CHECK_COMMAND}}",
        "{{MEMORY_HYGIENE_COMMAND}}",
        "post the same concise summary in chat",
    ],
    "REPO_MAP.md": [
        "meta_process.md",
    ],
    "AGENTS.md": [
        "Execution Process: process-tasks.md (canonical source",
        "Production Safety Gate: `dev/production-safety-gate.md`",
        "## Process Ownership (Avoid Drift)",
    ],
    "CLAUDE.md": [
        "Execution Process: process-tasks.md (canonical source",
        "Production Safety Gate: `dev/production-safety-gate.md`",
        "## Process Ownership (Avoid Drift)",
    ],
    ".agent/maps/meta_process.md": [
        "## Ownership Model",
        "dev/memory.md",
        "dev/task-summaries.md",
    ],
    "dev/memory.md": [
        "## Stable Preferences",
        "## Mistakes Log",
        "## Do Not Repeat Checklist",
    ],
    "dev/task-summaries.md": [
        "## Entry Template",
        "## Entries",
    ],
    "dev/production-safety-gate.md": [
        "## Required Checklist",
        "## Evidence",
    ],
    "dev/skills/route-context/SKILL.md": [
        "name: route-context",
    ],
    "dev/skills/execute-parent-task/SKILL.md": [
        "name: execute-parent-task",
    ],
    "dev/skills/run-quality-gates/SKILL.md": [
        "name: run-quality-gates",
    ],
    "dev/skills/update-memory-and-summary/SKILL.md": [
        "name: update-memory-and-summary",
    ],
    "dev/skills/release-safety-gate/SKILL.md": [
        "name: release-safety-gate",
    ],
    "scripts/check_acceptance_contracts.py": [
        "Acceptance contracts check passed.",
        "REQUIRED_FIELDS",
    ],
    "scripts/check_production_safety_gate.py": [
        "REQUIRED_CHECKLIST_ITEMS",
        "Production safety gate check passed",
    ],
    "scripts/check_memory_hygiene.py": [
        "Memory hygiene check passed.",
        "ALLOWED_STATUSES",
    ],
    ".agent/maps/testing_quality.md": [
        "## Gate Levels",
    ],
}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    failures: list[str] = []

    for relative_path, snippets in REQUIRED_SNIPPETS.items():
        file_path = root / relative_path
        if not file_path.exists():
            failures.append(f"Missing file: {relative_path}")
            continue

        text = file_path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in text:
                failures.append(
                    f"Missing snippet in {relative_path}: {snippet!r}",
                )

    if failures:
        print("Documentation policy consistency check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Documentation policy consistency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
