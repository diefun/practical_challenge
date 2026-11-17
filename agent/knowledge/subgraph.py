"""The centralized informational pipeline.

    START ─┬─> rag_retrieve ─┐
           └─> kg_query ─────┴─> synthesize ─> END

Both retrievers run in parallel for every informational query; the synthesizer combines
document evidence (RAG) with structured facts (KG) into one grounded, cited answer.
This graph is exposed to the supervisor as a single tool, `knowledge_research`.
It has no access to the MCP server and only reads data.
"""

from __future__ import annotations

import json
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from agent.config import get_config
from agent.knowledge import kg, rag
from agent.llm import make_llm


class KnowledgeState(TypedDict, total=False):
    question: str
    rag_queries: list[str]
    rag_hits: list[dict]
    kg_result: dict
    answer: str


REWRITE_PROMPT = """Split the user's question into 1-3 short, self-contained search queries for a document
search over contracts (MSAs), statements of work and company policies. Each query should cover ONE
topic and name the client or project it concerns. Reply with one query per line and nothing else."""


def rewrite_queries(question: str) -> list[str]:
    """Multi-query retrieval: a question that mixes topics ("Apollo electricians, given Northwind's
    certification and rate rules") is searched as several focused queries plus the original."""
    try:
        reply = make_llm(get_config()["llm"]["worker_effort"]).invoke(
            [SystemMessage(REWRITE_PROMPT), HumanMessage(question)])
        subs = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", q).strip() for q in reply.text.splitlines()]
        subs = [q for q in subs if len(q) > 3][:3]
    except Exception:
        subs = []  # retrieval must never fail because of the rewrite step
    return [question, *subs]


def rag_retrieve(state: KnowledgeState) -> KnowledgeState:
    queries = rewrite_queries(state["question"])
    try:
        return {"rag_queries": queries, "rag_hits": rag.search_many(queries)}
    except Exception as exc:
        return {"rag_queries": queries,
                "rag_hits": [{"text": f"(document search unavailable: {exc})", "source": "error"}]}


def kg_query(state: KnowledgeState) -> KnowledgeState:
    try:
        return {"kg_result": kg.query_graph(state["question"])}
    except Exception as exc:
        return {"kg_result": {"cypher": None, "rows": [], "error": str(exc)}}


SYNTH_PROMPT = """You answer questions for a contractor-staffing team using ONLY the evidence below.
Two sources are provided:
- DOCUMENTS (contracts, SOWs, policies) — cite as [D1], [D2], ...
- KNOWLEDGE GRAPH rows (live staffing data: contractors, skills, certifications, projects, assignments) — cite as [KG]

Combine both when relevant (e.g. check graph candidates against contract or policy rules).
If the evidence does not contain the answer, say what is missing instead of guessing.
Be concise. End with a "Sources:" line listing the documents you cited."""


def synthesize(state: KnowledgeState) -> KnowledgeState:
    docs = "\n\n".join(
        f"[D{i}] ({h.get('source')}, {h.get('section', '')})\n{h.get('text', '')}"
        for i, h in enumerate(state.get("rag_hits", []), 1)
    ) or "(none)"
    kgr = state.get("kg_result", {})
    graph = (
        f"Cypher: {kgr.get('cypher')}\nRows: {json.dumps(kgr.get('rows', []), default=str)}"
        + (f"\nError: {kgr['error']}" if kgr.get("error") else "")
    )
    llm = make_llm(get_config()["llm"]["worker_effort"])
    reply = llm.invoke([
        SystemMessage(SYNTH_PROMPT),
        HumanMessage(f"QUESTION: {state['question']}\n\nDOCUMENTS:\n{docs}\n\nKNOWLEDGE GRAPH:\n{graph}"),
    ])
    return {"answer": reply.text}


def build_knowledge_graph():
    g = StateGraph(KnowledgeState)
    g.add_node("rag_retrieve", rag_retrieve)
    g.add_node("kg_query", kg_query)
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "rag_retrieve")
    g.add_edge(START, "kg_query")
    g.add_edge(["rag_retrieve", "kg_query"], "synthesize")  # join: wait for both
    g.add_edge("synthesize", END)
    return g.compile(name="knowledge_research")


knowledge_graph = build_knowledge_graph()


@tool
async def knowledge_research(question: str) -> str:
    """Answer any informational question about contractors, skills, certifications, availability,
    projects, clients, assignments, contracts (MSAs/SOWs), rates and company policies.

    Searches the document corpus (RAG) and the staffing knowledge graph (KG) together and returns a
    cited answer. It only reads data and never changes anything. Pass a complete, self-contained
    question (resolve pronouns from the conversation)."""
    result = await knowledge_graph.ainvoke({"question": question})
    return result["answer"]
