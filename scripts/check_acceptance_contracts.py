#!/usr/bin/env python3
"""Validate implementation contract files.

This check is intentionally lightweight and markdown-structure-based.
It enforces that active contract docs include the sections required by the
repository process.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED_HEADINGS = (
    "## Status",
    "## Source Finding",
    "## Intent",
    "## Problem",
    "## Objective",
    "## Dependencies",
    "## Scope",
    "## Out of Scope",
    "## Invariants",
    "## Affected Files",
    "## Implementation Plan",
    "## Verification",
    "## Acceptance Criteria",
    "## Evidence Required",
    "## Notes for Future Sessions",
)


def iter_contract_files(contracts_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in contracts_dir.glob("*.md")
        if path.name != "README.md"
    )


def validate_contract_file(contract_file: Path) -> list[str]:
    if not contract_file.exists():
        return [f"Contract file not found: {contract_file}"]

    text = contract_file.read_text(encoding="utf-8")
    failures: list[str] = []
    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            failures.append(
                f"{contract_file}: missing required section {heading}",
            )
    return failures


def validate_contracts_dir(contracts_dir: Path) -> list[str]:
    if not contracts_dir.exists():
        return [f"Contracts directory not found: {contracts_dir}"]

    contract_files = iter_contract_files(contracts_dir)
    if not contract_files:
        return [f"No contract files found in: {contracts_dir}"]

    failures: list[str] = []
    for contract_file in contract_files:
        failures.extend(validate_contract_file(contract_file))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate implementation contract files.",
    )
    parser.add_argument(
        "--contracts-dir",
        default="dev/contracts",
        help="Directory containing active implementation contracts.",
    )
    parser.add_argument(
        "--contract-file",
        help="Validate one specific implementation contract file.",
    )
    args = parser.parse_args()

    if args.contract_file:
        failures = validate_contract_file(Path(args.contract_file))
    else:
        failures = validate_contracts_dir(Path(args.contracts_dir))

    if failures:
        print("Acceptance contracts check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Acceptance contracts check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
