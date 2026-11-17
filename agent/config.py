"""Loads config.yaml and applies environment-variable overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_ENV_OVERRIDES = {
    ("llm", "model"): "LLM_MODEL",
    ("neo4j", "uri"): "NEO4J_URI",
    ("neo4j", "user"): "NEO4J_USER",
    ("neo4j", "password"): "NEO4J_PASSWORD",
    ("qdrant", "url"): "QDRANT_URL",
    ("qdrant", "path"): "QDRANT_PATH",
    ("agent", "require_approval_for_destructive"): "REQUIRE_APPROVAL",
}


@lru_cache
def get_config() -> dict[str, Any]:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    for (section, key), env in _ENV_OVERRIDES.items():
        if (val := os.getenv(env)) is not None:
            if isinstance(cfg[section].get(key), bool):
                val = val.lower() in ("1", "true", "yes")
            cfg[section][key] = val
    # Per-server URL overrides: MCP_<SERVER_NAME>_URL
    for name, server in cfg.get("mcp_servers", {}).items():
        if url := os.getenv(f"MCP_{name.upper()}_URL"):
            server["url"] = url
    qpath = Path(cfg["qdrant"]["path"])
    if not qpath.is_absolute():
        cfg["qdrant"]["path"] = str(ROOT / qpath)
    return cfg
