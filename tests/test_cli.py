"""Tests for axis_core.cli."""

from __future__ import annotations

from pathlib import Path

import pytest

from axis_core import cli


class _FakeResult:
    def __init__(
        self,
        *,
        success: bool,
        output_raw: str = "",
        output: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.success = success
        self.output_raw = output_raw
        self.output = output
        self.error = error


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


@pytest.mark.unit
class TestCliRuntime:
    def test_ask_runs_single_prompt(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class _FakeAgent:
            def __init__(self) -> None:
                self.calls: list[tuple[str, float | None]] = []

            def run(self, prompt: str, timeout: float | None = None) -> _FakeResult:
                self.calls.append((prompt, timeout))
                return _FakeResult(success=True, output_raw="answer")

        fake_agent = _FakeAgent()
        monkeypatch.setattr(cli, "_make_agent", lambda _args: fake_agent)

        exit_code = cli.main(["ask", "What is axis-core?", "--timeout", "12"])

        assert exit_code == 0
        assert fake_agent.calls == [("What is axis-core?", 12.0)]
        assert capsys.readouterr().out.strip().endswith("answer")

    def test_ask_returns_error_on_failed_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class _FakeAgent:
            def run(self, _prompt: str, timeout: float | None = None) -> _FakeResult:
                return _FakeResult(
                    success=False,
                    error=RuntimeError("boom"),
                )

        monkeypatch.setattr(cli, "_make_agent", lambda _args: _FakeAgent())
        exit_code = cli.main(["ask", "test"])

        assert exit_code == 1
        assert "ask failed: boom" in capsys.readouterr().err

    def test_chat_runs_until_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class _FakeSession:
            id = "session-abc"

            def __init__(self) -> None:
                self.messages: list[str] = []

            def run(self, prompt: str, timeout: float | None = None) -> _FakeResult:
                self.messages.append(prompt)
                return _FakeResult(success=True, output_raw="pong")

        class _FakeAgent:
            def __init__(self, session: _FakeSession) -> None:
                self._session = session
                self.calls: list[tuple[str | None, int]] = []

            def session(
                self,
                id: str | None = None,
                *,
                max_history: int = 100,
            ) -> _FakeSession:
                self.calls.append((id, max_history))
                return self._session

        fake_session = _FakeSession()
        fake_agent = _FakeAgent(fake_session)
        monkeypatch.setattr(cli, "_make_agent", lambda _args: fake_agent)

        inputs = iter(["ping", "/session", "/help", "/exit"])
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

        exit_code = cli.main(["chat", "--session-id", "resume-1", "--max-history", "42"])

        output = capsys.readouterr().out
        assert exit_code == 0
        assert fake_agent.calls == [("resume-1", 42)]
        assert fake_session.messages == ["ping"]
        assert "assistant> pong" in output
        assert "session-abc" in output
        assert "Commands: /help, /exit, /quit, /session" in output

    def test_session_placeholder(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = cli.main(["session"])
        assert exit_code == 0
        assert "not implemented yet" in capsys.readouterr().out

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
