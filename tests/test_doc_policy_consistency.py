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
