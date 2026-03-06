from pathlib import Path

from scripts import check_acceptance_contracts


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_validate_contract_file_passes_with_required_sections(tmp_path: Path) -> None:
    contract_file = tmp_path / "dev/contracts/04-example.md"
    write_file(
        contract_file,
        "\n".join(
            [
                "# Contract 04: Example",
                "",
                *check_acceptance_contracts.REQUIRED_HEADINGS,
            ]
        ),
    )

    failures = check_acceptance_contracts.validate_contract_file(contract_file)

    assert failures == []


def test_validate_contract_file_reports_missing_sections(tmp_path: Path) -> None:
    contract_file = tmp_path / "dev/contracts/04-example.md"
    write_file(
        contract_file,
        "# Contract 04: Example\n\n## Status\n",
    )

    failures = check_acceptance_contracts.validate_contract_file(contract_file)

    assert "## Source Finding" in failures[0]
    assert len(failures) == len(check_acceptance_contracts.REQUIRED_HEADINGS) - 1


def test_validate_contracts_dir_ignores_readme_and_checks_contracts(tmp_path: Path) -> None:
    contracts_dir = tmp_path / "dev/contracts"
    write_file(contracts_dir / "README.md", "# Contracts\n")
    write_file(
        contracts_dir / "04-example.md",
        "\n".join(
            [
                "# Contract 04: Example",
                "",
                *check_acceptance_contracts.REQUIRED_HEADINGS,
            ]
        ),
    )

    failures = check_acceptance_contracts.validate_contracts_dir(contracts_dir)

    assert failures == []


def test_validate_contracts_dir_fails_when_no_contracts_exist(tmp_path: Path) -> None:
    contracts_dir = tmp_path / "dev/contracts"
    contracts_dir.mkdir(parents=True)

    failures = check_acceptance_contracts.validate_contracts_dir(contracts_dir)

    assert failures == [f"No contract files found in: {contracts_dir}"]
