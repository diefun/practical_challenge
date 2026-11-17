"""Load the CSV seed data into Neo4j as a knowledge graph (idempotent MERGEs)."""

from __future__ import annotations

import csv
from pathlib import Path

from neo4j import GraphDatabase

from agent.config import ROOT, get_config

CSV = ROOT / "data" / "csv"


def rows(name: str) -> list[dict]:
    with open(CSV / name, newline="") as f:
        return list(csv.DictReader(f))


STATEMENTS: list[tuple[str, str]] = [
    ("clients.csv", """
        UNWIND $rows AS r
        MERGE (c:Client {id: r.id}) SET c.name = r.name, c.industry = r.industry"""),
    ("contractors.csv", """
        UNWIND $rows AS r
        MERGE (c:Contractor {id: r.id})
        SET c.name = r.name, c.email = r.email, c.hourly_rate = toFloat(r.hourly_rate),
            c.available_from = date(r.available_from), c.status = r.status
        MERGE (l:Location {name: r.location})
        MERGE (c)-[:LOCATED_IN]->(l)"""),
    ("skills.csv", """
        UNWIND $rows AS r
        MATCH (c:Contractor {id: r.contractor_id})
        MERGE (s:Skill {name: r.skill})
        MERGE (c)-[h:HAS_SKILL]->(s) SET h.level = toInteger(r.level)"""),
    ("certifications.csv", """
        UNWIND $rows AS r
        MATCH (c:Contractor {id: r.contractor_id})
        MERGE (cert:Certification {name: r.certification})
        MERGE (c)-[h:HOLDS]->(cert) SET h.expires = date(r.expires)"""),
    ("projects.csv", """
        UNWIND $rows AS r
        MATCH (cl:Client {id: r.client_id})
        MERGE (p:Project {id: r.id})
        SET p.name = r.name, p.status = r.status, p.start_date = date(r.start_date),
            p.end_date = date(r.end_date), p.budget_usd = toInteger(r.budget_usd)
        MERGE (p)-[:FOR_CLIENT]->(cl)
        MERGE (l:Location {name: r.location})
        MERGE (p)-[:LOCATED_IN]->(l)"""),
    ("project_requirements.csv", """
        UNWIND $rows AS r
        MATCH (p:Project {id: r.project_id})
        MERGE (s:Skill {name: r.skill})
        MERGE (p)-[q:REQUIRES]->(s)
        SET q.min_level = toInteger(r.min_level), q.headcount = toInteger(r.headcount)"""),
    ("assignments.csv", """
        UNWIND $rows AS r
        MATCH (c:Contractor {id: r.contractor_id}), (p:Project {id: r.project_id})
        MERGE (c)-[a:ASSIGNED_TO]->(p)
        SET a.role = r.role, a.start_date = date(r.start_date), a.end_date = date(r.end_date)"""),
]

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Contractor) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Project) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Client) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Skill) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Certification) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Location) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:WorkOrder) REQUIRE n.id IS UNIQUE",
]


def build(reset: bool = True) -> dict[str, int]:
    cfg = get_config()["neo4j"]
    with GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"])) as driver:
        if reset:
            driver.execute_query("MATCH (n) DETACH DELETE n")
        for stmt in CONSTRAINTS:
            driver.execute_query(stmt)
        for file, stmt in STATEMENTS:
            driver.execute_query(stmt, rows=rows(file))
        records, _, _ = driver.execute_query(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label"
        )
        return {r["label"]: r["n"] for r in records}


if __name__ == "__main__":
    print(build())
