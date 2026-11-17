"""Project actions: status changes and timesheets."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from mcp_server.db import audit, resolve, write

ProjectStatus = Literal["planning", "active", "on_hold", "completed", "cancelled"]


def register(mcp: FastMCP) -> None:
    @mcp.tool(
        annotations=ToolAnnotations(
            title="Update project status",
            readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False,
        )
    )
    def update_project_status(
        project: Annotated[str, Field(description="Project id (e.g. 'P004') or full/partial name")],
        status: Annotated[ProjectStatus, Field(description="New project status")],
        note: Annotated[str, Field(description="Short reason for the change")] = "",
    ) -> dict[str, Any]:
        """Change a project's status (planning, active, on_hold, completed, cancelled)."""

        def tx_fn(tx):
            p = resolve(tx, "Project", project)
            old = tx.run(
                "MATCH (p:Project {id:$pid}) WITH p, p.status AS old SET p.status=$status, "
                "p.status_note=$note RETURN old",
                pid=p["id"], status=status, note=note,
            ).single()["old"]
            return {"status": "updated", "project": p, "from": old, "to": status, "note": note}

        result = write(tx_fn)
        audit("update_project_status", {"project": project, "status": status, "note": note}, result)
        return result

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Log timesheet hours",
            readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False,
        )
    )
    def log_timesheet(
        contractor: Annotated[str, Field(description="Contractor id or name")],
        project: Annotated[str, Field(description="Project id or name")],
        work_date: Annotated[date, Field(description="Day worked (YYYY-MM-DD)")],
        hours: Annotated[float, Field(gt=0, le=12, multiple_of=0.5, description="Hours worked, 0.5 steps, max 12")],
    ) -> dict[str, Any]:
        """Log hours a contractor worked on a project for one day. The contractor must be assigned to
        the project; entries older than 14 days or in the future are rejected."""
        today = date.today()
        if work_date > today:
            raise ToolError("Cannot log hours for a future date.")
        if work_date < today - timedelta(days=14):
            raise ToolError("Entries older than 14 days need finance approval.")

        def tx_fn(tx):
            c = resolve(tx, "Contractor", contractor)
            p = resolve(tx, "Project", project)
            ok = tx.run(
                "MATCH (:Contractor {id:$cid})-[a:ASSIGNED_TO]->(:Project {id:$pid}) RETURN count(a) AS n",
                cid=c["id"], pid=p["id"],
            ).single()["n"]
            if not ok:
                raise ToolError(f"{c['name']} is not assigned to {p['name']}.")
            tx.run(
                """MATCH (c:Contractor {id:$cid}), (p:Project {id:$pid})
                   CREATE (c)-[:LOGGED]->(t:TimesheetEntry {work_date: date($d), hours: $h})-[:FOR_PROJECT]->(p)""",
                cid=c["id"], pid=p["id"], d=work_date.isoformat(), h=hours,
            )
            return {"status": "logged", "contractor": c, "project": p,
                    "work_date": work_date.isoformat(), "hours": hours}

        result = write(tx_fn)
        audit("log_timesheet", {"contractor": contractor, "project": project,
                                "work_date": work_date, "hours": hours}, result)
        return result
