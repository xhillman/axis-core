#!/usr/bin/env python3
"""Validate key policy anchors across process and prompt docs.

This script is intentionally lightweight and string-based. It helps catch
accidental drift when updating agent/process guidance documents.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

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

PYPROJECT_VERSION_RE = re.compile(r'^version = "(?P<version>[^"]+)"$', re.MULTILINE)
BUILD_RELEASE_VERSION_RE = re.compile(
    r"^- Current version: `(?P<version>[^`]+)`$",
    re.MULTILINE,
)


def extract_pyproject_version(pyproject_text: str) -> str | None:
    match = PYPROJECT_VERSION_RE.search(pyproject_text)
    if match is None:
        return None
    return match.group("version")


def extract_build_release_version(build_release_text: str) -> str | None:
    match = BUILD_RELEASE_VERSION_RE.search(build_release_text)
    if match is None:
        return None
    return match.group("version")


def find_build_release_version_drift(root: Path) -> list[str]:
    failures: list[str] = []

    pyproject_path = root / "pyproject.toml"
    build_release_path = root / ".agent/maps/build_release.md"

    if not pyproject_path.exists():
        return ["Missing file: pyproject.toml"]
    if not build_release_path.exists():
        return ["Missing file: .agent/maps/build_release.md"]

    pyproject_version = extract_pyproject_version(pyproject_path.read_text(encoding="utf-8"))
    if pyproject_version is None:
        return ["Missing version in pyproject.toml"]

    build_release_version = extract_build_release_version(
        build_release_path.read_text(encoding="utf-8"),
    )
    if build_release_version is None:
        return ["Missing current version in .agent/maps/build_release.md"]

    if build_release_version != pyproject_version:
        failures.append(
            "Version drift: .agent/maps/build_release.md reports "
            f"`{build_release_version}` but pyproject.toml reports `{pyproject_version}`",
        )

    return failures


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

    failures.extend(find_build_release_version_drift(root))

    if failures:
        print("Documentation policy consistency check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Documentation policy consistency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
