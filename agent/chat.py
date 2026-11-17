"""Terminal chat UI (alternative to Agent Chat UI): python -m agent.chat"""

from __future__ import annotations

import asyncio
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import build_graph, get_tools


async def main() -> None:
    tools = await get_tools()
    graph = build_graph(tools, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("Staffing Copilot. Tools:", ", ".join(t.name for t in tools))
    print("Type 'exit' to quit.\n")
    payload = None
    while True:
        if payload is None:
            text = input("you> ").strip()
            if text in ("exit", "quit"):
                return
            payload = {"messages": [HumanMessage(text)]}
        seen = len((await graph.aget_state(config)).values.get("messages", []))
        result = await graph.ainvoke(payload, config)
        payload = None
        for msg in result["messages"][seen:]:
            if isinstance(msg, AIMessage):
                for call in msg.tool_calls:
                    print(f"  [tool call] {call['name']}({call['args']})")
                if msg.text:
                    print(f"bot> {msg.text}\n")
            elif isinstance(msg, ToolMessage) and msg.status == "error":
                print(f"  [tool error] {msg.content}")
        if interrupts := result.get("__interrupt__"):
            req = interrupts[0].value[0]
            ans = input(f"approve {req['action_request']}? [y/N] ").strip().lower()
            payload = Command(resume=[{"type": "accept"} if ans == "y" else {"type": "ignore"}])


if __name__ == "__main__":
    asyncio.run(main())
