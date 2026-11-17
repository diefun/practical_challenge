"""Routing: the model picks informational vs. action tools on its own.

Offline tests use a scripted model to prove the wiring (tool calls reach the knowledge subgraph
or the MCP server). Live tests (need ANTHROPIC_API_KEY + running MCP server) check that Claude
picks the right tool for each labeled prompt, and picks up a newly added MCP server's tool with
no change to agent code.
"""

import json
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.config import get_config
from agent.graph import build_graph, get_tools
from tests.conftest import needs_llm, needs_neo4j

EVAL = [json.loads(line) for line in Path(__file__).with_name("eval_set.jsonl").read_text().splitlines()]


def _port_open(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        return True
    except OSError:
        return False


needs_mcp = pytest.mark.skipif(not _port_open(8000), reason="MCP server not running on :8000")


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*messages):
    return ScriptedModel(messages=iter(messages))


def call(name, args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}])


def tool_calls(messages):
    return [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]


@needs_mcp
@needs_neo4j
async def test_action_tool_call_reaches_mcp_server(fresh_kg):
    tools = await get_tools()
    graph = build_graph(tools, llm=scripted(
        call("create_work_order", {"project": "Apollo", "title": "Test order", "description": "x"}),
        AIMessage("Done."),
    ))
    out = await graph.ainvoke({"messages": [HumanMessage("open a work order")]})
    result = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert result.name == "create_work_order" and "WO-" in result.text


async def test_knowledge_tool_is_a_single_entry_point(monkeypatch):
    """Informational calls go through the whole subgraph (both retrievers + synthesis)."""
    from agent.knowledge import subgraph

    seen = []
    monkeypatch.setattr(subgraph.rag, "search_many", lambda qs: seen.append("rag") or [{"text": "t", "source": "s"}])
    monkeypatch.setattr(subgraph.kg, "query_graph", lambda q: seen.append("kg") or {"cypher": "c", "rows": []})
    monkeypatch.setattr(subgraph, "make_llm", lambda *_: scripted(AIMessage("synthesized")))
    graph = build_graph([subgraph.knowledge_research], llm=scripted(
        call("knowledge_research", {"question": "who is on Apollo?"}), AIMessage("answer")))
    out = await graph.ainvoke({"messages": [HumanMessage("who is on Apollo?")]})
    assert sorted(seen) == ["kg", "rag"]
    assert next(m for m in out["messages"] if isinstance(m, ToolMessage)).text == "synthesized"


async def test_no_mcp_server_degrades_gracefully(monkeypatch):
    cfg = get_config()
    monkeypatch.setitem(cfg, "mcp_servers", {"down": {"transport": "streamable_http",
                                                       "url": "http://127.0.0.1:9/mcp"}})
    assert [t.name for t in await get_tools()] == ["knowledge_research"]


# ---------------------------------------------------------------- live (Claude) tests

@needs_llm
@needs_mcp
@needs_neo4j
@pytest.mark.parametrize("case", EVAL, ids=[c["prompt"][:40] for c in EVAL])
async def test_live_routing(case, fresh_kg):
    graph = build_graph(await get_tools())
    out = await graph.ainvoke({"messages": [HumanMessage(case["prompt"])]})
    calls = tool_calls(out["messages"])
    assert case["expect"] in calls, calls
    if case["expect"] == "knowledge_research":
        assert all(c == "knowledge_research" for c in calls), f"read-only question triggered an action: {calls}"


EXTRA_SERVER = '''
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("notifications", port=8765)

@mcp.tool()
def send_reminder(contractor: str, message: str) -> dict:
    """Send a text-message reminder to a contractor."""
    return {"status": "sent", "to": contractor, "message": message}

mcp.run(transport="streamable-http")
'''


@needs_llm
@needs_mcp
async def test_new_mcp_server_is_used_without_code_changes(monkeypatch, tmp_path):
    script = tmp_path / "extra_server.py"
    script.write_text(EXTRA_SERVER)
    proc = subprocess.Popen([sys.executable, str(script)])
    try:
        for _ in range(50):
            if _port_open(8765):
                break
            time.sleep(0.2)
        servers = dict(get_config()["mcp_servers"])
        servers["notifications"] = {"transport": "streamable_http", "url": "http://127.0.0.1:8765/mcp"}
        monkeypatch.setitem(get_config(), "mcp_servers", servers)  # == editing config.yaml
        graph = build_graph(await get_tools())
        out = await graph.ainvoke({"messages": [HumanMessage(
            "Text Ana Torres a reminder that her Apollo induction is Monday at 8am.")]})
        assert "send_reminder" in tool_calls(out["messages"])
    finally:
        proc.terminate()
