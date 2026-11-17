"""Write-side data access for the MCP action server.

The MCP server is the only component that writes to the staffing graph. It is deliberately
self-contained (no imports from `agent/`) so it can be deployed and scaled separately.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcp.server.fastmcp.exceptions import ToolError
from neo4j import Driver, GraphDatabase, ManagedTransaction

AUDIT_LOG = Path(os.getenv("MCP_AUDIT_LOG", Path(__file__).with_name("audit.log.jsonl")))


@lru_cache
def get_driver() -> Driver:
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "contractor-demo")),
    )


def write(fn, *args, **kwargs) -> Any:
    """Run `fn(tx, ...)` in a single write transaction (all-or-nothing)."""
    with get_driver().session() as session:
        return session.execute_write(fn, *args, **kwargs)


def resolve(tx: ManagedTransaction, label: str, ref: str) -> dict:
    """Find a node by id or name, case-insensitive. Match tiers, best first: exact id/name, a word
    of the name starting with `ref` ("Ana" -> "Ana Torres"), then any substring.
    Errors if nothing matches or the best tier is ambiguous."""
    ref = ref.strip()
    low = ref.lower()
    rows = tx.run(
        f"""MATCH (n:{label}) WHERE n.id = $ref OR toLower(n.name) CONTAINS $low
            RETURN n.id AS id, n.name AS name""",
        ref=ref, low=low,
    ).data()
    tiers = (
        [r for r in rows if r["id"] == ref or r["name"].lower() == low],
        [r for r in rows if any(w.startswith(low) for w in r["name"].lower().split())
         or r["name"].lower().startswith(low)],
        rows,
    )
    matches = next((t for t in tiers if t), [])
    if not matches:
        raise ToolError(f"No {label.lower()} matches '{ref}'.")
    if len(matches) > 1:
        names = ", ".join(f"{r['name']} ({r['id']})" for r in matches)
        raise ToolError(f"'{ref}' is ambiguous; did you mean one of: {names}?")
    return {"id": matches[0]["id"], "name": matches[0]["name"]}


def audit(tool: str, args: dict, result: dict) -> None:
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "tool": tool, "args": args, "result": result}
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
