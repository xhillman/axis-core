"""Tests for axis_core.cli."""

from __future__ import annotations

from pathlib import Path

import pytest

from axis_core import cli


@pytest.mark.unit
class TestCliInit:
    def test_init_writes_env_for_synaptic_and_bootstraps(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / ".env"
        db_path = tmp_path / "data" / "synaptic.db"
        bootstrapped: list[str] = []

        monkeypatch.setattr(cli, "_module_available", lambda _name: True)
        monkeypatch.setattr(
            cli,
            "_bootstrap_synaptic_db",
            lambda path: bootstrapped.append(path),
        )

        exit_code = cli.main(
            [
                "init",
                "--yes",
                "--memory",
                "synaptic",
                "--planner",
                "auto",
                "--env-file",
                str(env_file),
                "--synaptic-db-path",
                str(db_path),
                "--bootstrap-synaptic-db",
            ]
        )

        assert exit_code == 0
        assert bootstrapped == [str(db_path)]
        content = env_file.read_text(encoding="utf-8")
        assert "AXIS_DEFAULT_MEMORY=synaptic" in content
        assert "AXIS_DEFAULT_PLANNER=auto" in content
        assert f"AXIS_SYNAPTIC_PATH={db_path}" in content

    def test_init_installs_missing_memory_dependency(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / ".env"
        installs: list[str] = []
        availability = {"synaptic_core": False}

        def fake_available(module_name: str) -> bool:
            if module_name == "synaptic_core":
                return availability["synaptic_core"]
            return True

        def fake_install(requirement: str) -> None:
            installs.append(requirement)
            availability["synaptic_core"] = True

        monkeypatch.setattr(cli, "_module_available", fake_available)
        monkeypatch.setattr(cli, "_install_requirement", fake_install)

        exit_code = cli.main(
            [
                "init",
                "--yes",
                "--memory",
                "synaptic",
                "--planner",
                "auto",
                "--env-file",
                str(env_file),
                "--install-missing",
            ]
        )

        assert exit_code == 0
        assert installs == ["synaptic-core>=0.1.1"]
        content = env_file.read_text(encoding="utf-8")
        assert "AXIS_DEFAULT_MEMORY=synaptic" in content

    def test_init_fails_if_missing_memory_dependency_and_not_installing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / ".env"
        monkeypatch.setattr(
            cli,
            "_module_available",
            lambda module_name: False if module_name == "synaptic_core" else True,
        )

        exit_code = cli.main(
            [
                "init",
                "--yes",
                "--memory",
                "synaptic",
                "--planner",
                "auto",
                "--env-file",
                str(env_file),
            ]
        )

        assert exit_code == 1
        assert not env_file.exists()

    def test_init_installs_requested_bundle(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / ".env"
        installs: list[str] = []
        availability = {"synaptic_core": False}

        def fake_available(module_name: str) -> bool:
            if module_name == "synaptic_core":
                return availability["synaptic_core"]
            return True

        def fake_install(requirement: str) -> None:
            installs.append(requirement)
            if requirement == "synaptic-core>=0.1.1":
                availability["synaptic_core"] = True

        monkeypatch.setattr(cli, "_module_available", fake_available)
        monkeypatch.setattr(cli, "_install_requirement", fake_install)

        exit_code = cli.main(
            [
                "init",
                "--yes",
                "--install",
                "synaptic",
                "--memory",
                "ephemeral",
                "--planner",
                "auto",
                "--env-file",
                str(env_file),
            ]
        )

        assert exit_code == 0
        assert installs == ["synaptic-core>=0.1.1"]
