"""MCP server: schemas, annotations and action behaviour (in-memory MCP client, real Neo4j)."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_server.server import mcp
from tests.conftest import needs_neo4j


async def call(name: str, args: dict):
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        return await session.call_tool(name, args)


async def test_tools_have_valid_schemas_and_annotations():
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
    assert {"assign_contractor", "unassign_contractor", "update_project_status",
            "create_work_order", "log_timesheet"} <= tools.keys()
    for t in tools.values():
        assert t.description and t.inputSchema["type"] == "object"
        assert t.annotations is not None and t.annotations.readOnlyHint is False
    assert tools["unassign_contractor"].annotations.destructiveHint is True
    assert set(tools["assign_contractor"].inputSchema["required"]) == {"contractor", "project", "role", "start_date"}
    assert tools["update_project_status"].inputSchema["properties"]["status"]["enum"] == [
        "planning", "active", "on_hold", "completed", "cancelled"]


@needs_neo4j
async def test_assign_then_unassign(fresh_kg):
    res = await call("assign_contractor", {"contractor": "Ana", "project": "Apollo",
                                           "role": "Electrician", "start_date": "2026-11-24"})
    assert not res.isError, res.content
    data = res.structuredContent
    assert data["contractor"]["id"] == "C001" and data["project"]["id"] == "P001"
    assert data["end_date"] == "2027-03-31"  # defaults to project end

    from agent.knowledge.kg import run_read_only  # verify via the read-only KG path
    rows = run_read_only("MATCH (:Contractor {id:'C001'})-[a:ASSIGNED_TO]->(p:Project) RETURN p.name AS p")
    assert rows == [{"p": "Apollo"}]

    res = await call("unassign_contractor", {"contractor": "C001", "project": "P001", "reason": "test"})
    assert not res.isError


@needs_neo4j
@pytest.mark.parametrize("args, message", [
    ({"contractor": "Ana", "project": "Northwind Clinic", "role": "x", "start_date": "2027-02-01"}, "on_hold"),
    ({"contractor": "Ana", "project": "Apollo", "role": "x", "start_date": "2026-11-18"}, "not available until"),
    ({"contractor": "Ben", "project": "Apollo", "role": "x", "start_date": "2026-12-16"}, None),
    ({"contractor": "Diego", "project": "Helios", "role": "x", "start_date": "2027-01-06"}, None),
    ({"contractor": "Grace", "project": "Apollo", "role": "x", "start_date": "2026-12-01",
      "end_date": "2026-12-02"}, None),
    ({"contractor": "Zed", "project": "Apollo", "role": "x", "start_date": "2026-12-01"}, "No contractor"),
])
async def test_assign_business_rules(fresh_kg, args, message):
    res = await call("assign_contractor", args)
    if message is None:
        assert not res.isError, res.content
    else:
        assert res.isError and message in res.content[0].text


@needs_neo4j
async def test_overlapping_assignment_rejected(fresh_kg):
    # Diego is on Apollo until 2027-01-04 and available from 2027-01-05.
    first = await call("assign_contractor", {"contractor": "Diego", "project": "Helios",
                                             "role": "PM", "start_date": "2027-01-05"})
    assert not first.isError
    second = await call("assign_contractor", {"contractor": "Diego", "project": "Bluepeak HQ",
                                              "role": "PM", "start_date": "2027-02-01"})
    assert second.isError and "already on" in second.content[0].text


@needs_neo4j
async def test_status_work_order_and_ambiguity(fresh_kg):
    res = await call("update_project_status", {"project": "P004", "status": "active", "note": "variance approved"})
    assert res.structuredContent["from"] == "on_hold"
    res = await call("create_work_order", {"project": "Apollo", "title": "Fix conduit",
                                           "description": "Re-route conduit in MRI suite", "priority": "high"})
    assert res.structuredContent["work_order_id"].startswith("WO-")
    res = await call("create_work_order", {"project": "o", "title": "x" * 5, "description": "d"})
    assert res.isError and "ambiguous" in res.content[0].text  # matches several project names
    bad = await call("update_project_status", {"project": "P001", "status": "paused"})
    assert bad.isError  # schema enum enforced


@needs_neo4j
async def test_audit_log_written(fresh_kg):
    from mcp_server.db import AUDIT_LOG

    before = AUDIT_LOG.read_text().count("\n") if AUDIT_LOG.exists() else 0
    await call("update_project_status", {"project": "P002", "status": "active"})
    lines = AUDIT_LOG.read_text().splitlines()
    assert len(lines) == before + 1 and json.loads(lines[-1])["tool"] == "update_project_status"
