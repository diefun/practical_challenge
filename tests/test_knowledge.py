"""Informational pipeline: RAG retrieval, read-only KG access, subgraph shape."""

import pytest

from agent.knowledge import kg, rag
from agent.knowledge.subgraph import knowledge_graph
from tests.conftest import needs_llm, needs_neo4j


def test_subgraph_runs_rag_and_kg_in_parallel_then_synthesizes():
    g = knowledge_graph.get_graph()
    edges = {(e.source, e.target) for e in g.edges}
    assert {("__start__", "rag_retrieve"), ("__start__", "kg_query"),
            ("rag_retrieve", "synthesize"), ("kg_query", "synthesize"), ("synthesize", "__end__")} == edges


@pytest.mark.parametrize("question, expected_source", [
    ("What is the not-to-exceed rate for electricians at Bluepeak Energy?", "msa_bluepeak_energy.md"),
    ("How much notice is required to remove a contractor from a Northwind project?", "msa_northwind_health.md"),
    ("When is the MRI suite shielding sign-off milestone?", "sow_apollo.md"),
    ("What priority should a safety incident work order have?", "policy_safety.md"),
])
def test_rag_retrieves_relevant_document(question, expected_source):
    sources = [h["source"] for h in rag.search(question, top_k=3)]
    assert expected_source in sources


@needs_neo4j
def test_kg_read_query():
    rows = kg.run_read_only(
        "MATCH (c:Contractor)-[:HOLDS]->(:Certification {name:'OSHA 30'}) RETURN c.name AS n ORDER BY n")
    assert [r["n"] for r in rows] == ["Ana Torres", "Farid Haddad", "Jamal Wright"]


@needs_neo4j
@pytest.mark.parametrize("cypher", [
    "MATCH (c:Contractor) SET c.status = 'hacked'",
    "CREATE (:Contractor {id:'X'})",
    "MATCH (n) DETACH DELETE n",
])
def test_kg_path_rejects_writes(cypher):
    with pytest.raises(PermissionError):
        kg.run_read_only(cypher)


@needs_neo4j
@needs_llm
async def test_knowledge_pipeline_combines_rag_and_kg():
    out = await knowledge_graph.ainvoke({
        "question": "Which available contractors could fill the electrician roles on Apollo, "
                    "given Northwind's certification and rate rules?"})
    assert out["kg_result"]["rows"], out["kg_result"]
    assert any(h["source"] == "msa_northwind_health.md" for h in out["rag_hits"])
    assert "Ana Torres" in out["answer"]
