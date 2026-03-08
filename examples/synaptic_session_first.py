"""Canonical Synaptic 0.3 session-first usage with axis-core interop."""

from __future__ import annotations

import asyncio
from pathlib import Path

from synaptic_core import Synaptic

from axis_core.adapters.memory.synaptic import SynapticMemory
from axis_core.session import Message, Session


def _embedding_fn(text: str) -> list[float]:
    """Small deterministic embedding function for local demos."""
    seed = sum(ord(ch) for ch in text)
    return [float((seed + idx * 17) % 101) / 100.0 for idx in range(8)]


async def main() -> None:
    db_path = Path("synaptic_example.db")

    # Canonical Synaptic 0.3 API (session-first surface).
    client = Synaptic(db_path=str(db_path), embedding_fn=_embedding_fn)
    session_client = client.session("demo-session")

    await session_client.remember("Deploy status: green in us-east-1.")
    recall = await session_client.recall("deploy status", top_k=3)
    print(f"session recall nodes: {len(recall.nodes)}")

    await client.set("ops:last_deploy", {"region": "us-east-1", "status": "green"})
    value = await client.get("ops:last_deploy")
    print(f"set/get value: {value}")

    rows = await client.find("ops:last", limit=5)
    print(f"find rows: {len(rows)}")

    # Axis adapter maps protocol store/retrieve/search to canonical client methods.
    memory = SynapticMemory(db_path=str(db_path), embedding_fn=_embedding_fn)
    await memory.store("axis:checkpoint", {"ok": True}, metadata={"source": "example"})
    retrieved = await memory.retrieve("axis:checkpoint")
    print(f"axis adapter retrieve: {retrieved}")

    axis_session = Session(id="axis-demo-session")
    axis_session.add_message(Message(role="user", content="Hello, axis."))
    stored_session = await memory.store_session(axis_session)
    print(f"stored session version: {stored_session.version}")


if __name__ == "__main__":
    asyncio.run(main())
