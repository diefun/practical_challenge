"""Contractor Ops MCP server — the only way the system performs actions.

Run:  python -m mcp_server.server            (streamable HTTP on :8000/mcp)
      python -m mcp_server.server --stdio    (for VS Code / Copilot / Claude Desktop)

To add an action, add a tool in mcp_server/tools/ and register it below. The agent finds it
on its next run; no change to agent code is needed.
"""

from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

from mcp_server.tools import assignments, projects, work_orders

INSTRUCTIONS = (
    "Action tools for the contractor staffing system. Every tool changes data. "
    "Contractors and projects can be referenced by id or name."
)


def build_server() -> FastMCP:
    mcp = FastMCP(
        "contractor-ops",
        instructions=INSTRUCTIONS,
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8000")),
    )
    for module in (assignments, projects, work_orders):
        module.register(mcp)
    return mcp


mcp = build_server()

if __name__ == "__main__":
    mcp.run(transport="stdio" if "--stdio" in sys.argv else "streamable-http")
