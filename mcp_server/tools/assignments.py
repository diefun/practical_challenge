"""Assignment actions: assign / unassign contractors."""

from __future__ import annotations

from datetime import date
from typing import Any, Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from mcp_server.db import audit, resolve, write

ContractorRef = Annotated[str, Field(description="Contractor id (e.g. 'C001') or full/partial name")]
ProjectRef = Annotated[str, Field(description="Project id (e.g. 'P001') or full/partial name")]


def register(mcp: FastMCP) -> None:
    @mcp.tool(
        annotations=ToolAnnotations(
            title="Assign contractor to project",
            readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False,
        )
    )
    def assign_contractor(
        contractor: ContractorRef,
        project: ProjectRef,
        role: Annotated[str, Field(description="Role on the project, e.g. 'Electrician'")],
        start_date: Annotated[date, Field(description="First day of the assignment (YYYY-MM-DD)")],
        end_date: Annotated[date | None, Field(description="Last day; defaults to the project end date")] = None,
    ) -> dict[str, Any]:
        """Assign a contractor to a project in the staffing system.

        Checks that the project accepts staff (not on hold, completed or cancelled), the contractor is
        available by the start date, and the new assignment does not overlap an existing one."""

        def tx_fn(tx):
            c = resolve(tx, "Contractor", contractor)
            p = resolve(tx, "Project", project)
            info = tx.run(
                """MATCH (c:Contractor {id:$cid}), (p:Project {id:$pid})
                   RETURN c.available_from AS avail, p.status AS status, p.end_date AS pend""",
                cid=c["id"], pid=p["id"],
            ).single()
            if info["status"] in ("on_hold", "completed", "cancelled"):
                raise ToolError(f"{p['name']} is {info['status']}; contractors cannot be assigned.")
            end = end_date or info["pend"].to_native()
            if end < start_date:
                raise ToolError("end_date is before start_date.")
            if info["avail"] and start_date < info["avail"].to_native():
                raise ToolError(f"{c['name']} is not available until {info['avail']}.")
            clash = tx.run(
                """MATCH (:Contractor {id:$cid})-[a:ASSIGNED_TO]->(other:Project)
                   WHERE a.start_date <= date($end) AND a.end_date >= date($start)
                   RETURN other.name AS project, a.start_date AS s, a.end_date AS e""",
                cid=c["id"], start=start_date.isoformat(), end=end.isoformat(),
            ).data()
            if clash:
                o = clash[0]
                raise ToolError(f"{c['name']} is already on {o['project']} from {o['s']} to {o['e']}.")
            tx.run(
                """MATCH (c:Contractor {id:$cid}), (p:Project {id:$pid})
                   MERGE (c)-[a:ASSIGNED_TO]->(p)
                   SET a.role=$role, a.start_date=date($start), a.end_date=date($end), c.status='assigned'""",
                cid=c["id"], pid=p["id"], role=role, start=start_date.isoformat(), end=end.isoformat(),
            )
            return {"status": "assigned", "contractor": c, "project": p, "role": role,
                    "start_date": start_date.isoformat(), "end_date": end.isoformat()}

        result = write(tx_fn)
        audit("assign_contractor", {"contractor": contractor, "project": project, "role": role,
                                    "start_date": start_date, "end_date": end_date}, result)
        return result

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Remove contractor from project",
            readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False,
        )
    )
    def unassign_contractor(
        contractor: ContractorRef,
        project: ProjectRef,
        reason: Annotated[str, Field(description="Why the contractor is being removed")],
    ) -> dict[str, Any]:
        """Remove a contractor's assignment from a project. Marks the contractor available again when
        they have no other assignments."""

        def tx_fn(tx):
            c = resolve(tx, "Contractor", contractor)
            p = resolve(tx, "Project", project)
            deleted = tx.run(
                """MATCH (:Contractor {id:$cid})-[a:ASSIGNED_TO]->(:Project {id:$pid})
                   DELETE a RETURN count(*) AS n""",
                cid=c["id"], pid=p["id"],
            ).single()["n"]
            if not deleted:
                raise ToolError(f"{c['name']} is not assigned to {p['name']}.")
            tx.run(
                """MATCH (c:Contractor {id:$cid}) WHERE NOT (c)-[:ASSIGNED_TO]->()
                   SET c.status='available', c.available_from=date()""",
                cid=c["id"],
            )
            return {"status": "unassigned", "contractor": c, "project": p, "reason": reason}

        result = write(tx_fn)
        audit("unassign_contractor", {"contractor": contractor, "project": project, "reason": reason}, result)
        return result

