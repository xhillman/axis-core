#!/usr/bin/env python3
"""Validate a few high-value documentation drift risks.

This checker stays intentionally small:
- `REPO_MAP.md` router links must resolve
- a curated set of metadata path claims must still exist
- the maintained version in `build_release.md` must match `pyproject.toml`
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
PYPROJECT_VERSION_RE = re.compile(r'^version = "(?P<version>[^"]+)"$', re.MULTILINE)
BUILD_RELEASE_VERSION_RE = re.compile(
    r"^- Current version: `(?P<version>[^`]+)`$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class PathClaim:
    document_path: str
    reference_text: str
    target_path: str


PATH_CLAIMS: tuple[PathClaim, ...] = (
    PathClaim(
        document_path="REPO_MAP.md",
        reference_text="context/",
        target_path="axis_core/context/",
    ),
    PathClaim(
        document_path="REPO_MAP.md",
        reference_text="engine/lifecycle.py",
        target_path="axis_core/engine/lifecycle.py",
    ),
    PathClaim(
        document_path="REPO_MAP.md",
        reference_text="axis_core/adapters/",
        target_path="axis_core/adapters/",
    ),
    PathClaim(
        document_path="REPO_MAP.md",
        reference_text="tests/",
        target_path="tests/",
    ),
    PathClaim(
        document_path=".agent/maps/meta_process.md",
        reference_text="dev/process-tasks.md",
        target_path="dev/process-tasks.md",
    ),
    PathClaim(
        document_path=".agent/maps/meta_process.md",
        reference_text="dev/spec-driven.md",
        target_path="dev/spec-driven.md",
    ),
    PathClaim(
        document_path=".agent/maps/meta_process.md",
        reference_text="dev/contracts/README.md",
        target_path="dev/contracts/README.md",
    ),
    PathClaim(
        document_path=".agent/maps/meta_process.md",
        reference_text="dev/archive/",
        target_path="dev/archive/",
    ),
    PathClaim(
        document_path="dev/process-tasks.md",
        reference_text="python3 scripts/check_doc_policy_consistency.py",
        target_path="scripts/check_doc_policy_consistency.py",
    ),
    PathClaim(
        document_path="AGENTS.md",
        reference_text="python3 scripts/check_doc_policy_consistency.py",
        target_path="scripts/check_doc_policy_consistency.py",
    ),
    PathClaim(
        document_path="CLAUDE.md",
        reference_text="python3 scripts/check_doc_policy_consistency.py",
        target_path="scripts/check_doc_policy_consistency.py",
    ),
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


def iter_relative_markdown_links(markdown_text: str) -> Iterable[tuple[str, str]]:
    for label, target in MARKDOWN_LINK_RE.findall(markdown_text):
        if target.startswith(("http://", "https://", "mailto:", "#", "/")):
            continue
        yield label, target


def path_target_exists(root: Path, target_path: str) -> bool:
    if any(token in target_path for token in ("*", "?", "[")):
        return any(root.glob(target_path))
    return (root / target_path).exists()


def find_router_link_drift(root: Path, router_path: str = "REPO_MAP.md") -> list[str]:
    file_path = root / router_path
    if not file_path.exists():
        return [f"Missing file: {router_path}"]

    failures: list[str] = []
    router_text = file_path.read_text(encoding="utf-8")
    for label, target in iter_relative_markdown_links(router_text):
        resolved_target = (file_path.parent / target).resolve().relative_to(root.resolve())
        if not (root / resolved_target).exists():
            failures.append(
                "Router link drift in "
                f"{router_path}: link '{label}' points to missing path "
                f"{resolved_target.as_posix()}",
            )
    return failures


def find_path_claim_drift(root: Path, claims: Iterable[PathClaim] = PATH_CLAIMS) -> list[str]:
    failures: list[str] = []

    for claim in claims:
        file_path = root / claim.document_path
        if not file_path.exists():
            failures.append(f"Missing file: {claim.document_path}")
            continue

        text = file_path.read_text(encoding="utf-8")
        if claim.reference_text not in text:
            failures.append(
                "Missing metadata claim in "
                f"{claim.document_path}: {claim.reference_text!r}",
            )
            continue

        if not path_target_exists(root, claim.target_path):
            failures.append(
                "Metadata drift in "
                f"{claim.document_path}: referenced path {claim.target_path!r} does not exist",
            )

    return failures


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
    failures = [
        *find_router_link_drift(root),
        *find_path_claim_drift(root),
        *find_build_release_version_drift(root),
    ]

    if failures:
        print("Documentation policy consistency check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Documentation policy consistency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
