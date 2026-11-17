import os
import socket

import pytest

from agent.config import get_config


def _neo4j_up() -> bool:
    uri = get_config()["neo4j"]["uri"].split("//")[1]
    host, port = uri.split(":")
    try:
        socket.create_connection((host, int(port)), timeout=1).close()
        return True
    except OSError:
        return False


needs_neo4j = pytest.mark.skipif(not _neo4j_up(), reason="Neo4j not running (docker compose up -d neo4j)")
needs_llm = pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")


@pytest.fixture
def fresh_kg():
    """Reset the knowledge graph to the CSV seed state."""
    from ingest.build_kg import build

    build()
    yield
    build()
