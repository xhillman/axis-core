"""Initialize phase: create RunContext, validate config."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from axis_core.attachments import AttachmentLike
from axis_core.budget import Budget
from axis_core.cancel import CancelToken
from axis_core.context import NormalizedInput, RunContext, RunState
from axis_core.errors import ConfigError

if TYPE_CHECKING:
    from axis_core.engine.lifecycle import LifecycleEngine

logger = logging.getLogger("axis_core.engine")


async def initialize(
    engine: LifecycleEngine,
    input_text: str,
    agent_id: str,
    budget: Budget,
    context: dict[str, Any] | None = None,
    attachments: list[AttachmentLike] | None = None,
    cancel_token: CancelToken | None = None,
    config: Any | None = None,
) -> RunContext:
    """Initialize phase: create RunContext, validate config.

    Args:
        engine: The lifecycle engine instance
        input_text: User input text
        agent_id: Agent identifier
        budget: Budget limits for this run
        context: Optional context dict for sharing state
        attachments: Optional list of attachments
        cancel_token: Optional cancellation token
        config: Optional resolved configuration

    Returns:
        Initialized RunContext

    Raises:
        ConfigError: If input is empty or config is invalid
    """
    if not input_text or not input_text.strip():
        raise ConfigError(message="Input must not be empty")

    run_id = str(uuid.uuid4())

    normalized_input = NormalizedInput(
        text=input_text.strip(),
        original=input_text,
    )

    ctx = RunContext(
        run_id=run_id,
        agent_id=agent_id,
        input=normalized_input,
        context=context or {},
        attachments=attachments or [],
        config=config,
        budget=budget,
        state=RunState(),
        trace=None,
        started_at=datetime.utcnow(),
        cycle_count=0,
        cancel_token=cancel_token,
    )

    from axis_core.engine.lifecycle import Phase

    await engine._emit(
        "phase_entered",
        run_id=run_id,
        phase=Phase.INITIALIZE.value,
        data={"agent_id": agent_id, "input_length": len(input_text)},
    )

    logger.debug("Initialized run %s for agent %s", run_id, agent_id)

    await engine._emit(
        "phase_exited",
        run_id=run_id,
        phase=Phase.INITIALIZE.value,
    )

    return ctx
