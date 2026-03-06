"""Public package export regression tests."""

import importlib


class TestPackageExports:
    """Public package exports should remain stable across submodule imports."""

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
