"""Test that dependency lockfile exists and is valid for supply chain security (MED-2)."""

import re
from pathlib import Path

import pytest


class TestDependencyLockfile:
    """Verify lockfile exists and contains expected dependencies."""

    @pytest.fixture
    def repo_root(self) -> Path:
        """Get repository root directory."""
        # Test file is in tests/, so go up one level
        return Path(__file__).parent.parent

    @pytest.fixture
    def lockfile_path(self, repo_root: Path) -> Path:
        """Get expected lockfile path."""
        return repo_root / "requirements.lock"

    @pytest.fixture
    def pyproject_path(self, repo_root: Path) -> Path:
        """Get project metadata file path."""
        return repo_root / "pyproject.toml"

    @pytest.fixture
    def readme_path(self, repo_root: Path) -> Path:
        """Get README path for install/documentation checks."""
        return repo_root / "README.md"

    def test_lockfile_exists(self, lockfile_path: Path) -> None:
        """Verify that a dependency lockfile exists in the repository root."""
        assert lockfile_path.exists(), (
            f"Lockfile not found at {lockfile_path}. "
            "Run: uv pip compile pyproject.toml -o requirements.lock"
        )

    def test_lockfile_contains_core_dependencies(self, lockfile_path: Path) -> None:
        """Verify lockfile contains all core dependencies with pinned versions."""
        content = lockfile_path.read_text()

        # Core dependencies from pyproject.toml
        required_deps = ["pydantic", "python-dotenv", "httpx"]

        for dep in required_deps:
            # Match pattern like "pydantic==2.5.3" (exact version pin)
            pattern = rf"^{dep}==[\d.]+$"
            assert re.search(pattern, content, re.MULTILINE), (
                f"Dependency '{dep}' not found with exact version pin in lockfile. "
                f"Expected pattern: {pattern}"
            )

    def test_lockfile_format_is_valid(self, lockfile_path: Path) -> None:
        """Verify lockfile follows requirements.txt format with exact pins."""
        content = lockfile_path.read_text()
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        # Verify we have at least some dependencies
        assert len(lines) >= 3, "Lockfile should contain at least core dependencies"

        # Verify all non-comment lines are exact pins (==) not ranges (>=, ~=, etc.)
        for line in lines:
            # Skip lines that are just package names with extras like "pydantic[email]"
            if "[" in line and "]" in line:
                continue
            # Each line should have == for exact version pinning
            assert "==" in line, f"Line '{line}' should use exact version pin (==), not ranges"

    def test_lockfile_is_not_empty(self, lockfile_path: Path) -> None:
        """Verify lockfile is not empty and contains content."""
        content = lockfile_path.read_text()
        assert len(content) > 0, "Lockfile should not be empty"
        assert content.strip(), "Lockfile should not be just whitespace"

    def test_optional_provider_extras_swap_ollama_for_openrouter(
        self,
        pyproject_path: Path,
    ) -> None:
        """Verify provider extras match supported provider scope."""
        content = pyproject_path.read_text()

        assert re.search(r'^openrouter\s*=\s*\["openai>=1.0"\]$', content, re.MULTILINE), (
            "Expected openrouter extra backed by OpenAI SDK."
        )
        assert not re.search(r"^ollama\s*=", content, re.MULTILINE), (
            "Ollama extra should not be declared in current provider scope."
        )
        assert '"axis-core[openrouter]"' in content, (
            "Full extra should include openrouter optional install target."
        )
        assert '"axis-core[ollama]"' not in content, (
            "Full extra should not include deprecated ollama optional install target."
        )

    def test_readme_documents_openrouter_optional_install(self, readme_path: Path) -> None:
        """Verify README includes OpenRouter install guidance."""
        content = readme_path.read_text()
        assert "OpenRouter" in content
        assert "axis-core[openrouter]" in content
