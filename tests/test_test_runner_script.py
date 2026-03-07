from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest


@pytest.mark.unit
def test_wrapper_disables_plugin_autoload_and_loads_required_plugins(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_script = repo_root / "scripts" / "test.sh"

    wrapper_path = tmp_path / "scripts" / "test.sh"
    wrapper_path.parent.mkdir(parents=True)
    wrapper_path.write_text(source_script.read_text())
    wrapper_path.chmod(source_script.stat().st_mode | stat.S_IXUSR)

    fake_python = tmp_path / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "out_dir=${FAKE_OUTPUT_DIR:?}\n"
        "printf '%s\\n' \"${PYTEST_DISABLE_PLUGIN_AUTOLOAD-}\" > \"$out_dir/env.txt\"\n"
        "printf '%s\\n' \"$@\" > \"$out_dir/args.txt\"\n"
    )
    fake_python.chmod(0o755)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    env = os.environ.copy()
    env["FAKE_OUTPUT_DIR"] = str(output_dir)

    completed = subprocess.run(
        [str(wrapper_path), "tests/test_lockfile.py", "--cov=axis_core"],
        check=False,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "env.txt").read_text().strip() == "1"
    assert (output_dir / "args.txt").read_text().splitlines() == [
        "-m",
        "pytest",
        "-p",
        "pytest_asyncio.plugin",
        "-p",
        "pytest_cov",
        "tests/test_lockfile.py",
        "--cov=axis_core",
    ]
