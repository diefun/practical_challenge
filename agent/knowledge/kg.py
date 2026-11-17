"""Knowledge-graph retrieval: text-to-Cypher over Neo4j, read-only."""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from neo4j import READ_ACCESS, Driver, GraphDatabase

from agent.config import get_config
from agent.llm import make_llm

SCHEMA = """
Node labels and properties:
  (:Contractor {id, name, email, hourly_rate: float, available_from: date, status: 'available'|'assigned'})
  (:Skill {name})            e.g. Electrical, Plumbing, HVAC, Solar Installation, Project Management,
                             Structural Engineering, BIM Modeling, Carpentry, Drywall, Concrete, Safety Inspection
  (:Certification {name})    e.g. 'OSHA 10', 'OSHA 30', PMP, 'PE License', 'EPA 608', 'NABCEP PV Installer',
                             'Master Electrician License', 'Journeyman Plumber License', CSP
  (:Project {id, name, status: 'planning'|'active'|'on_hold'|'completed'|'cancelled',
             start_date: date, end_date: date, budget_usd: int})
  (:Client {id, name, industry})
  (:Location {name})         Austin, Denver, Phoenix, Remote
  (:WorkOrder {id, title, description, priority, status, created_at})
  (:TimesheetEntry {work_date: date, hours: float})

Relationships:
  (:Contractor)-[:HAS_SKILL {level: 1-5}]->(:Skill)
  (:Contractor)-[:HOLDS {expires: date}]->(:Certification)
  (:Contractor)-[:LOCATED_IN]->(:Location)
  (:Contractor)-[:ASSIGNED_TO {role, start_date: date, end_date: date}]->(:Project)
  (:Contractor)-[:LOGGED]->(:TimesheetEntry)-[:FOR_PROJECT]->(:Project)
  (:Project)-[:REQUIRES {min_level: int, headcount: int}]->(:Skill)
  (:Project)-[:FOR_CLIENT]->(:Client)
  (:Project)-[:LOCATED_IN]->(:Location)
  (:Project)-[:HAS_WORK_ORDER]->(:WorkOrder)
"""

FEW_SHOTS = """
Q: Who is assigned to Apollo?
MATCH (c:Contractor)-[a:ASSIGNED_TO]->(p:Project) WHERE toLower(p.name) = 'apollo'
RETURN c.name AS contractor, a.role AS role, a.start_date AS start, a.end_date AS end

Q: Which available electricians with level 4+ hold a valid OSHA 30?
MATCH (c:Contractor {status:'available'})-[h:HAS_SKILL]->(:Skill {name:'Electrical'}) WHERE h.level >= 4
MATCH (c)-[x:HOLDS]->(:Certification {name:'OSHA 30'}) WHERE x.expires >= date()
RETURN c.name, h.level, c.hourly_rate, c.available_from, x.expires

Q: Which skills does Helios Solar Farm still need?
MATCH (p:Project)-[r:REQUIRES]->(s:Skill) WHERE toLower(p.name) CONTAINS 'helios'
OPTIONAL MATCH (c:Contractor)-[:ASSIGNED_TO]->(p) WHERE (c)-[:HAS_SKILL]->(s)
RETURN s.name AS skill, r.min_level, r.headcount, count(c) AS assigned

Q: Whose certifications expire in the next 60 days?
MATCH (c:Contractor)-[h:HOLDS]->(cert:Certification) WHERE h.expires <= date() + duration({days:60})
RETURN c.name, cert.name, h.expires ORDER BY h.expires

Q: Which contractors in Phoenix are free before December?
MATCH (c:Contractor)-[:LOCATED_IN]->(:Location {name:'Phoenix'}) WHERE c.available_from < date('2026-12-01')
RETURN c.name, c.available_from, c.hourly_rate

Q: Projects for Northwind Health and their status
MATCH (p:Project)-[:FOR_CLIENT]->(cl:Client) WHERE cl.name CONTAINS 'Northwind'
RETURN p.name, p.status, p.start_date, p.end_date

Q: Open work orders on Apollo
MATCH (p:Project)-[:HAS_WORK_ORDER]->(w:WorkOrder) WHERE toLower(p.name)='apollo' AND w.status='open'
RETURN w.id, w.title, w.priority, w.created_at

Q: Best candidates for the Plumbing requirement on Apollo
MATCH (p:Project {name:'Apollo'})-[r:REQUIRES]->(s:Skill {name:'Plumbing'})
MATCH (c:Contractor)-[h:HAS_SKILL]->(s) WHERE h.level >= r.min_level
OPTIONAL MATCH (c)-[:LOCATED_IN]->(l:Location)
RETURN c.name, h.level, c.status, c.available_from, c.hourly_rate, l.name AS location
"""

PROMPT = """You translate questions about a contractor-staffing business into ONE read-only Cypher query.
Today's date is {today}.

{schema}

Examples:
{shots}

Rules:
- Use only labels, relationships and properties from the schema.
- Read-only: MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT. Never write.
- Match names case-insensitively (toLower / CONTAINS) unless the exact value is known.
- Return human-readable fields (names, dates, levels), LIMIT 50.
- If the graph cannot answer the question at all (e.g. it is purely about contract wording or policy),
  reply with exactly: NO_QUERY
Reply with only the Cypher in a ```cypher block, or NO_QUERY."""


@lru_cache
def get_driver() -> Driver:
    cfg = get_config()["neo4j"]
    return GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


def run_read_only(cypher: str, limit: int = 50) -> list[dict]:
    """Execute Cypher only after Neo4j's planner confirms it is a read query."""
    with get_driver().session(default_access_mode=READ_ACCESS) as session:
        query_type = session.run(f"EXPLAIN {cypher}").consume().query_type
        if query_type != "r":
            raise PermissionError(f"Refusing non-read query (type={query_type!r}).")
        rows = session.execute_read(lambda tx: tx.run(cypher).data())
    return json.loads(json.dumps(rows[:limit], default=str))


def _extract_cypher(text: str) -> str | None:
    if "NO_QUERY" in text:
        return None
    m = re.search(r"```(?:cypher)?\s*(.+?)```", text, re.S)
    return (m.group(1) if m else text).strip().rstrip(";")


def query_graph(question: str, max_attempts: int = 3) -> dict:
    """Generate Cypher for the question, run it read-only, retry with error feedback."""
    llm = make_llm(get_config()["llm"]["worker_effort"])
    messages = [
        SystemMessage(PROMPT.format(today=date.today().isoformat(), schema=SCHEMA, shots=FEW_SHOTS)),
        HumanMessage(question),
    ]
    cypher, error = None, None
    for _ in range(max_attempts):
        reply = llm.invoke(messages)
        cypher = _extract_cypher(reply.text)
        if cypher is None:
            return {"cypher": None, "rows": [], "error": None}
        try:
            return {"cypher": cypher, "rows": run_read_only(cypher), "error": None}
        except Exception as exc:  # syntax error, non-read query, etc. -> let the model fix it
            error = str(exc)
            messages += [reply, HumanMessage(f"That query failed: {error}\nReturn a corrected query.")]
    return {"cypher": cypher, "rows": [], "error": error}
