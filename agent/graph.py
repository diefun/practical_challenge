"""Supervisor agent.

    START -> agent --(LLM emitted tool calls?)--> tools -> agent ... -> END

The LLM alone decides which tool to call. Its tools are:
  * `knowledge_research`: the centralized RAG + KG subgraph (informational route)
  * every tool discovered from the configured MCP servers (action route)
There is no keyword matching or intent if/else: the only branch is LangGraph's standard
`tools_condition` ("did the model request a tool?").
"""

from __future__ import annotations

from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import get_config
from agent.knowledge.subgraph import knowledge_research
from agent.llm import make_llm
from agent.mcp_tools import is_destructive, load_mcp_tools, with_approval
from agent.prompts import SUPERVISOR_PROMPT


def build_graph(tools: list[BaseTool], llm: BaseChatModel | None = None, checkpointer=None):
    model = (llm or make_llm()).bind_tools(tools)

    async def agent(state: MessagesState):
        system = SystemMessage(SUPERVISOR_PROMPT.format(today=date.today().isoformat()))
        return {"messages": [await model.ainvoke([system, *state["messages"]])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer, name="staffing_copilot")


async def get_tools() -> list[BaseTool]:
    tools: list[BaseTool] = [knowledge_research, *await load_mcp_tools()]
    if get_config()["agent"]["require_approval_for_destructive"]:
        tools = [with_approval(t) if is_destructive(t) else t for t in tools]
    return tools


async def make_graph():
    """Graph factory used by `langgraph dev` (see langgraph.json). Tools are discovered per build."""
    return build_graph(await get_tools())
