"""Discover action tools from every MCP server listed in config.yaml.

The agent never hard-codes action tools: whatever the MCP servers advertise via `tools/list`
is converted into LangChain tools. Add a tool to a server (or a server to config.yaml) and
the agent can use it on its next run.
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import interrupt

from agent.config import get_config

log = logging.getLogger(__name__)


async def load_mcp_tools() -> list[BaseTool]:
    servers = {name: dict(spec) for name, spec in get_config().get("mcp_servers", {}).items()}
    client = MultiServerMCPClient(servers)
    tools: list[BaseTool] = []
    for name in servers:
        try:
            tools += await client.get_tools(server_name=name)
        except Exception as exc:  # one unreachable server must not take the agent down
            log.warning("MCP server %r unavailable, its tools are skipped: %s", name, exc)
    return tools


def is_destructive(tool: BaseTool) -> bool:
    """Read the MCP `destructiveHint` annotation (exposed by the adapter as tool metadata)."""
    return bool((tool.metadata or {}).get("destructiveHint"))


def with_approval(tool: BaseTool) -> BaseTool:
    """Wrap a tool so it pauses for human approval (Agent Chat UI interrupt) before running."""

    async def _run(**kwargs):
        request = {
            "action_request": {"action": tool.name, "args": kwargs},
            "config": {"allow_accept": True, "allow_edit": True, "allow_respond": True, "allow_ignore": True},
            "description": f"Approve `{tool.name}`? {tool.description.splitlines()[0]}",
        }
        response = interrupt([request])[0]
        kind = response.get("type")
        if kind == "edit":
            kwargs = response["args"]["args"]
        elif kind != "accept":
            note = response.get("args") if kind == "response" else ""
            return f"The user did not approve {tool.name}. {note}".strip()
        return await tool.ainvoke(kwargs)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_run,
        metadata=tool.metadata,
    )
