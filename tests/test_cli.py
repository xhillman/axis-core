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
        assert installs == ["axis-core[synaptic]"]

    def test_init_yes_defaults_to_synaptic_and_auto_installs_dependency(
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
            if requirement == "axis-core[synaptic]":
                availability["synaptic_core"] = True

        monkeypatch.setattr(cli, "_module_available", fake_available)
        monkeypatch.setattr(cli, "_install_requirement", fake_install)

        exit_code = cli.main(
            [
                "init",
                "--yes",
                "--planner",
                "auto",
                "--env-file",
                str(env_file),
            ]
        )

        assert exit_code == 0
        assert installs == ["axis-core[synaptic]"]
        content = env_file.read_text(encoding="utf-8")
        assert "AXIS_DEFAULT_MEMORY=synaptic" in content


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
            if requirement == "axis-core[synaptic]":
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
        assert installs == ["axis-core[synaptic]"]

    def test_make_agent_and_metadata_uses_project_factory_with_selective_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _FakeAgent:
            def __init__(self, model: str | None) -> None:
                self._model = model
                self._planner = "project-planner"
                self._memory = "project-memory"
                self._system = "project-system"

            def run(self, _prompt: str, timeout: float | None = None) -> _FakeResult:
                return _FakeResult(success=True, output_raw="ok")

            def session(self, id: str | None = None, *, max_history: int = 100) -> object:
                return object()

        calls: list[tuple[str | None]] = []

        def _factory(*, model: str | None = None) -> _FakeAgent:
            calls.append((model,))
            return _FakeAgent(model=model)

        monkeypatch.setattr(
            cli,
            "_resolve_project_agent_reference",
            lambda _args: ("main:create_agent", "/tmp/project/pyproject.toml"),
        )
        monkeypatch.setattr(cli, "_load_object_from_ref", lambda _ref: _factory)

        args = cli.build_parser().parse_args(
            ["ask", "hello", "--model", "gpt-4o-mini", "--planner", "auto"]
        )
        agent, metadata = cli._make_agent_and_metadata(args)

        assert isinstance(agent, _FakeAgent)
        assert calls == [("gpt-4o-mini",)]
        assert metadata["mode"] == "project"
        assert metadata["reference"] == "main:create_agent"
        assert metadata["source"] == "/tmp/project/pyproject.toml"
        assert metadata["runtime_overrides"] == ["model", "planner"]
        assert metadata["dropped_overrides"] == ["planner"]

    def test_resolve_project_agent_reference_prefers_from_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            "[tool.axis_core]\nagent = \"main:create_agent\"\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        args = cli.build_parser().parse_args(["chat", "--from", "alt:factory"])
        ref, source = cli._resolve_project_agent_reference(args)

        assert ref == "alt:factory"
        assert source == "--from"

    def test_resolve_project_agent_reference_ignores_project_when_standalone(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            "[tool.axis_core]\nagent = \"main:create_agent\"\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        args = cli.build_parser().parse_args(["chat", "--standalone"])
        ref, source = cli._resolve_project_agent_reference(args)

        assert ref is None
        assert source is None

    def test_doctor_reports_resolved_agent_configuration(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        class _FakeAgent:
            _model = "gpt-4o-mini"
            _planner = "auto"
            _memory = "sqlite"
            _system = "Be concise."

            def run(self, _prompt: str, timeout: float | None = None) -> _FakeResult:
                return _FakeResult(success=True, output_raw="ok")

            def session(self, id: str | None = None, *, max_history: int = 100) -> object:
                return object()

        monkeypatch.setattr(
            cli,
            "_make_agent_and_metadata",
            lambda _args: (
                _FakeAgent(),
                {
                    "mode": "project",
                    "source": "/tmp/project/pyproject.toml",
                    "reference": "main:create_agent",
                    "runtime_overrides": ["model"],
                    "dropped_overrides": ["model"],
                },
            ),
        )

        exit_code = cli.main(["doctor", "--model", "override"])

        output = capsys.readouterr().out
        assert exit_code == 0
        assert "Axis Core CLI Doctor" in output
        assert "agent mode: project" in output
        assert "agent source: /tmp/project/pyproject.toml" in output
        assert "agent reference: main:create_agent" in output
        assert "runtime overrides: model" in output
        assert "dropped overrides: model" in output
        assert "model: 'gpt-4o-mini'" in output
