#!/usr/bin/env python3
"""Validate production safety gate checklist structure and completion status."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_CHECKLIST_ITEMS = (
    "Rollback plan defined and tested",
    "Data migration safety plan validated (forward and rollback)",
    "Runtime protections configured (timeouts, retries, rate limits, circuit breakers where applicable)",
    "Observability coverage confirmed (logs, metrics, traces, alerts)",
    "Security review completed (secrets handling, authz/authn, dependency risk)",
    "Performance/load validation completed for expected traffic profile",
    "Backup and restore path verified (if stateful components impacted)",
    "Incident response runbook updated for this release",
    "On-call and stakeholder communication plan confirmed",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate production safety gate markdown.",
    )
    parser.add_argument(
        "--gate-file",
        default="dev/production-safety-gate.md",
        help="Path to production safety gate file.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require all checklist items to be marked complete ([x]).",
    )
    args = parser.parse_args()

    gate_file = Path(args.gate_file)
    if not gate_file.exists():
        print(f"Production safety gate file not found: {gate_file}")
        return 1

    text = gate_file.read_text(encoding="utf-8")
    failures: list[str] = []

    for item in REQUIRED_CHECKLIST_ITEMS:
        if item not in text:
            failures.append(f"Missing required checklist item: {item}")
            continue

        unchecked = f"- [ ] {item}"
        checked = f"- [x] {item}"
        if args.require_complete and checked not in text:
            failures.append(f"Checklist item not complete: {item}")
        if not args.require_complete and unchecked not in text and checked not in text:
            failures.append(f"Checklist item missing checkbox format: {item}")

    if failures:
        print("Production safety gate check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    if args.require_complete:
        print("Production safety gate check passed (all required items complete).")
    else:
        print("Production safety gate check passed (structure valid).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
