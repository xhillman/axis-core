from pathlib import Path

from scripts import check_doc_policy_consistency


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_release_version_drift_is_reported(tmp_path: Path) -> None:
    write_file(
        tmp_path / "pyproject.toml",
        '[project]\nname = "axis-core"\nversion = "0.12.1"\n',
    )
    write_file(
        tmp_path / ".agent/maps/build_release.md",
        "# Build, CI & Release Map\n\n- Current version: `0.4.1`\n",
    )

    failures = check_doc_policy_consistency.find_build_release_version_drift(tmp_path)

    assert failures == [
        "Version drift: .agent/maps/build_release.md reports `0.4.1` but pyproject.toml "
        "reports `0.12.1`"
    ]


def test_build_release_version_check_passes_when_versions_match(tmp_path: Path) -> None:
    write_file(
        tmp_path / "pyproject.toml",
        '[project]\nname = "axis-core"\nversion = "0.12.1"\n',
    )
    write_file(
        tmp_path / ".agent/maps/build_release.md",
        "# Build, CI & Release Map\n\n- Current version: `0.12.1`\n",
    )

    failures = check_doc_policy_consistency.find_build_release_version_drift(tmp_path)

    assert failures == []


def test_router_link_drift_is_reported_for_missing_target(tmp_path: Path) -> None:
    write_file(
        tmp_path / "REPO_MAP.md",
        "| Task | Open |\n|---|---|\n| Docs | [meta_process.md](.agent/maps/meta_process.md) |\n",
    )

    failures = check_doc_policy_consistency.find_router_link_drift(tmp_path)

    assert failures == [
        "Router link drift in REPO_MAP.md: link 'meta_process.md' points to missing path "
        ".agent/maps/meta_process.md"
    ]


def test_path_claim_drift_is_reported_for_missing_target(tmp_path: Path) -> None:
    write_file(
        tmp_path / "REPO_MAP.md",
        "This router documents `axis_core/context/` as a live package.\n",
    )

    failures = check_doc_policy_consistency.find_path_claim_drift(
        tmp_path,
        claims=(
            check_doc_policy_consistency.PathClaim(
                document_path="REPO_MAP.md",
                reference_text="axis_core/context/",
                target_path="axis_core/context/",
            ),
        ),
    )

    assert failures == [
        "Metadata drift in REPO_MAP.md: referenced path 'axis_core/context/' does not exist",
    ]


def test_path_claim_drift_is_reported_for_missing_claim_text(tmp_path: Path) -> None:
    write_file(
        tmp_path / "REPO_MAP.md",
        "This router no longer names the expected package.\n",
    )
    write_file(tmp_path / "axis_core/context/__init__.py", "")

    failures = check_doc_policy_consistency.find_path_claim_drift(
        tmp_path,
        claims=(
            check_doc_policy_consistency.PathClaim(
                document_path="REPO_MAP.md",
                reference_text="axis_core/context/",
                target_path="axis_core/context/",
            ),
        ),
    )

    assert failures == [
        "Missing metadata claim in REPO_MAP.md: 'axis_core/context/'",
    ]


def test_ownership_claim_drift_is_reported_for_missing_summary_text(tmp_path: Path) -> None:
    write_file(tmp_path / "AGENTS.md", "## Process Ownership\n")

    failures = check_doc_policy_consistency.find_ownership_claim_drift(
        tmp_path,
        claims=(
            check_doc_policy_consistency.OwnershipClaim(
                document_path="AGENTS.md",
                reference_text=(
                    "- Keep behavioral prompt constraints in `dev/spec-driven.md` "
                    "only.\n"
                ),
            ),
        ),
    )

    assert failures == [
        "Ownership drift in AGENTS.md: missing claim "
        "'- Keep behavioral prompt constraints in `dev/spec-driven.md` only.\\n'",
    ]


def test_suite_metadata_drift_is_reported_for_missing_expected_paths(tmp_path: Path) -> None:
    write_file(
        tmp_path / "REPO_MAP.md",
        "tests/                   # Mirrors runtime areas and includes doc-policy, acceptance, "
        "and tooling checks\n",
    )
    write_file(tmp_path / "tests/test_doc_policy_consistency.py", "")

    failures = check_doc_policy_consistency.find_suite_metadata_drift(
        tmp_path,
        claims=(
            check_doc_policy_consistency.SuiteMetadataClaim(
                document_path="REPO_MAP.md",
                reference_text="doc-policy, acceptance, and tooling checks",
                expected_paths=(
                    "tests/test_doc_policy_consistency.py",
                    "tests/test_acceptance_contracts.py",
                    "tests/test_test_runner_script.py",
                ),
            ),
        ),
    )

    assert failures == [
        "Suite metadata drift in REPO_MAP.md: claim "
        "'doc-policy, acceptance, and tooling checks' expects existing paths "
        "['tests/test_acceptance_contracts.py', 'tests/test_test_runner_script.py']",
    ]


def test_suite_metadata_check_passes_when_expected_paths_exist(tmp_path: Path) -> None:
    write_file(
        tmp_path / ".agent/maps/testing_quality.md",
        "├── test_doc_policy_consistency.py         # Doc-policy checker coverage\n"
        "├── test_acceptance_contracts.py           # Contract-shape checker coverage\n"
        "├── test_test_runner_script.py             # `./scripts/test.sh` wrapper behavior\n",
    )
    write_file(tmp_path / "tests/test_doc_policy_consistency.py", "")
    write_file(tmp_path / "tests/test_acceptance_contracts.py", "")
    write_file(tmp_path / "tests/test_test_runner_script.py", "")

    failures = check_doc_policy_consistency.find_suite_metadata_drift(
        tmp_path,
        claims=(
            check_doc_policy_consistency.SuiteMetadataClaim(
                document_path=".agent/maps/testing_quality.md",
                reference_text="test_doc_policy_consistency.py",
                expected_paths=("tests/test_doc_policy_consistency.py",),
            ),
            check_doc_policy_consistency.SuiteMetadataClaim(
                document_path=".agent/maps/testing_quality.md",
                reference_text="test_acceptance_contracts.py",
                expected_paths=("tests/test_acceptance_contracts.py",),
            ),
            check_doc_policy_consistency.SuiteMetadataClaim(
                document_path=".agent/maps/testing_quality.md",
                reference_text="test_test_runner_script.py",
                expected_paths=("tests/test_test_runner_script.py",),
            ),
        ),
    )

    assert failures == []
