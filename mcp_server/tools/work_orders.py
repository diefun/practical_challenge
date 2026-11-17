"""Work-order actions."""

from __future__ import annotations

import uuid
from typing import Any, Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from mcp_server.db import audit, resolve, write


def register(mcp: FastMCP) -> None:
    @mcp.tool(
        annotations=ToolAnnotations(
            title="Create work order",
            readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False,
        )
    )
    def create_work_order(
        project: Annotated[str, Field(description="Project id or name")],
        title: Annotated[str, Field(min_length=3, max_length=120, description="Short title")],
        description: Annotated[str, Field(description="What needs to happen / expected outcome")],
        priority: Annotated[
            Literal["low", "medium", "high", "critical"],
            Field(description="critical = safety incidents/outages; high = blocks a milestone"),
        ] = "medium",
    ) -> dict[str, Any]:
        """Open a work order (task, defect or incident) on a project."""
        wo_id = f"WO-{uuid.uuid4().hex[:6].upper()}"

        def tx_fn(tx):
            p = resolve(tx, "Project", project)
            tx.run(
                """MATCH (p:Project {id:$pid})
                   CREATE (p)-[:HAS_WORK_ORDER]->(:WorkOrder {id:$id, title:$title, description:$desc,
                          priority:$priority, status:'open', created_at: datetime()})""",
                pid=p["id"], id=wo_id, title=title, desc=description, priority=priority,
            )
            return {"status": "created", "work_order_id": wo_id, "project": p,
                    "title": title, "priority": priority}

        result = write(tx_fn)
        audit("create_work_order", {"project": project, "title": title, "priority": priority}, result)
        return result
