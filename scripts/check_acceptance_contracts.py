#!/usr/bin/env python3
"""Validate acceptance contracts for open parent tasks in a task list.

This check is intentionally lightweight and markdown-structure-based.
It enforces that each open parent task has a complete acceptance contract.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


OPEN_PARENT_RE = re.compile(r"^- \[ \] (?P<task_id>\d+\.0)\b")
CONTRACT_HEADER_RE = re.compile(r"^### Task (?P<task_id>\d+\.\d+)$")

REQUIRED_FIELDS = (
    "- Behavior:",
    "- Negative Cases:",
    "- Non-Functional Constraints:",
    "- Verification:",
    "- Evidence Required:",
    "- Out of Scope:",
)


def find_open_parent_tasks(lines: list[str]) -> list[str]:
    tasks: list[str] = []
    for line in lines:
        match = OPEN_PARENT_RE.match(line)
        if match:
            tasks.append(match.group("task_id"))
    return tasks


def parse_acceptance_contracts(lines: list[str]) -> dict[str, list[str]]:
    contracts: dict[str, list[str]] = {}
    current_task: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        header_match = CONTRACT_HEADER_RE.match(line.strip())
        if header_match:
            current_task = header_match.group("task_id")
            contracts[current_task] = []
            continue

        if current_task is None:
            continue

        if line.startswith("## ") and not line.startswith("### "):
            current_task = None
            continue

        contracts[current_task].append(line)

    return contracts


def validate(open_tasks: list[str], contracts: dict[str, list[str]]) -> list[str]:
    failures: list[str] = []

    for task_id in open_tasks:
        if task_id not in contracts:
            failures.append(f"Missing acceptance contract for open parent task {task_id}")
            continue

        contract_blob = "\n".join(contracts[task_id])
        for field in REQUIRED_FIELDS:
            if field not in contract_blob:
                failures.append(
                    f"Task {task_id} missing acceptance field: {field}",
                )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate acceptance contracts for open parent tasks.",
    )
    parser.add_argument(
        "--tasks-file",
        default="dev/tasks-axis-core-prd.md",
        help="Path to task list markdown file.",
    )
    args = parser.parse_args()

    task_file = Path(args.tasks_file)
    if not task_file.exists():
        print(f"Task file not found: {task_file}")
        return 1

    lines = task_file.read_text(encoding="utf-8").splitlines()

    open_tasks = find_open_parent_tasks(lines)
    if not open_tasks:
        print("No open parent tasks found; acceptance contracts check passed.")
        return 0

    contracts = parse_acceptance_contracts(lines)
    failures = validate(open_tasks, contracts)

    if failures:
        print("Acceptance contracts check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Acceptance contracts check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
