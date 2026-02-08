#!/usr/bin/env python3
"""Validate assistant memory hygiene rules."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ALLOWED_STATUSES = {"Active", "Superseded", "Archived"}

# Conservative patterns for likely secrets/tokens.
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (RSA|EC|OPENSSH|PRIVATE) KEY-----"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate memory hygiene constraints.")
    parser.add_argument(
        "--memory-file",
        default="dev/memory.md",
        help="Path to assistant memory markdown file.",
    )
    parser.add_argument(
        "--max-checklist-items",
        type=int,
        default=7,
        help="Maximum allowed do-not-repeat checklist items.",
    )
    args = parser.parse_args()

    memory_file = Path(args.memory_file)
    if not memory_file.exists():
        print(f"Memory file not found: {memory_file}")
        return 1

    text = memory_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    failures: list[str] = []

    # Secret scan.
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            failures.append(f"Potential secret detected (pattern: {pattern.pattern})")

    # Mistake status scan.
    for line in lines:
        if not line.startswith("|"):
            continue
        # Mistakes rows include 5 columns and date in first cell.
        if re.match(r"^\|\s*20\d{2}-\d{2}-\d{2}\s*\|", line):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) >= 5:
                status = cells[-1]
                if status not in ALLOWED_STATUSES:
                    failures.append(
                        f"Invalid mistake status '{status}'. Allowed: {sorted(ALLOWED_STATUSES)}",
                    )

    # Do-not-repeat checklist length check.
    checklist_items = 0
    in_checklist = False
    for line in lines:
        if line.strip() == "## Do Not Repeat Checklist":
            in_checklist = True
            continue
        if in_checklist and line.startswith("## "):
            in_checklist = False
        if in_checklist and line.strip().startswith("-"):
            checklist_items += 1

    if checklist_items > args.max_checklist_items:
        failures.append(
            f"Too many checklist items ({checklist_items}); max is {args.max_checklist_items}",
        )

    if failures:
        print("Memory hygiene check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Memory hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
