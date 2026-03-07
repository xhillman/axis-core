"""Public package export regression tests."""

import importlib
import subprocess
import sys
from pathlib import Path


def _assert_package_import_does_not_call_dotenv(import_statement: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = "\n".join(
        [
            "import sys",
            "import types",
            "dotenv = types.ModuleType('dotenv')",
            "def load_dotenv(*args, **kwargs):",
            "    raise AssertionError('load_dotenv should not be called during package import')",
            "dotenv.load_dotenv = load_dotenv",
            "sys.modules['dotenv'] = dotenv",
            import_statement,
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


class TestPackageExports:
    """Public package exports should remain stable across submodule imports."""

    def test_package_import_does_not_load_dotenv(self) -> None:
        _assert_package_import_does_not_call_dotenv("import axis_core")

    def test_config_export_remains_singleton_after_submodule_import(self) -> None:
        import axis_core

        config_module = importlib.import_module("axis_core.config")
        pkg_config = axis_core.config
        direct_config = config_module.config

        assert isinstance(pkg_config, config_module.Config)
        assert pkg_config is direct_config
        assert axis_core.config is direct_config

    def test_tool_export_remains_decorator_after_submodule_import(self) -> None:
        import axis_core

        tool_module = importlib.import_module("axis_core.tool")
        pkg_tool = axis_core.tool
        direct_tool = tool_module.tool

        assert callable(pkg_tool)
        assert pkg_tool is direct_tool
        assert axis_core.tool is direct_tool
