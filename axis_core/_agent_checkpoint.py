from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from axis_core.errors import ConfigError


def checkpoint_path(checkpoint_dir: str, run_id: str) -> str:
    """Build a checkpoint file path for a run ID."""
    safe_run_id = run_id.replace("/", "_")
    return os.path.join(checkpoint_dir, f"{safe_run_id}.json")


def write_checkpoint_file(path: str, payload: dict[str, Any]) -> None:
    """Atomically write checkpoint payload to disk."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
    os.replace(tmp_path, path)


async def persist_checkpoint(checkpoint_dir: str, payload: dict[str, Any]) -> None:
    """Persist a checkpoint envelope to disk."""
    context = payload.get("context")
    if not isinstance(context, dict):
        raise ConfigError(message="Checkpoint context is missing or corrupt.")

    run_id = context.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ConfigError(message="Checkpoint context is missing run_id.")

    await asyncio.to_thread(
        write_checkpoint_file,
        checkpoint_path(checkpoint_dir, run_id),
        payload,
    )


def load_checkpoint_payload(checkpoint: str | dict[str, Any]) -> dict[str, Any]:
    """Load a checkpoint envelope from dict payload or JSON file path."""
    if isinstance(checkpoint, dict):
        return checkpoint

    try:
        with open(checkpoint, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(message=f"Checkpoint file not found: {checkpoint}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            message=f"Checkpoint file is not valid JSON: {checkpoint}",
            cause=exc,
        ) from exc
    except OSError as exc:
        raise ConfigError(
            message=f"Failed to read checkpoint file: {checkpoint}",
            cause=exc,
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError(message="Checkpoint payload must be a JSON object.")
    return data
