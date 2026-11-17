# Staffing Copilot: RAG + Knowledge Graph agent with MCP actions

A contractor-assignment chatbot. It answers questions by combining **documents (RAG)** and a
**knowledge graph (KG)**, and carries out changes **only through an MCP server**. Claude
(`claude-sonnet-5-5`) chooses the route by calling tools. There is no keyword routing.

```
Agent Chat UI / terminal ──► LangGraph server (agent/graph.py)
                               supervisor: agent ⇄ ToolNode   (only branch: tools_condition)
                               │
       informational route     ├─ knowledge_research  ── LangGraph subgraph (agent/knowledge/)
                               │     START ─┬─► rag_retrieve  (multi-query → Qdrant hybrid: dense + BM25, RRF)
                               │            └─► kg_query      (text→Cypher, read-only Neo4j)
                               │                   └──► synthesize (cited answer) ─► END
                               │
       action route            └─ <every tool discovered over MCP>  ──HTTP──► mcp_server/ (FastMCP)
                                     assign_contractor · unassign_contractor · update_project_status
                                     create_work_order · log_timesheet                    │
                                                                                Neo4j (writes)
```

## How it meets the requirements

| Requirement | Where |
|---|---|
| Informational queries go through one RAG + KG flow | `agent/knowledge/subgraph.py`: every call runs both retrievers in parallel, then synthesizes |
| Actions only through an MCP server | `mcp_server/`: separate process/container, the only component with write access |
| Model chooses the route, no if/else or keyword logic | `agent/graph.py`: the LLM picks tools; the graph only checks *whether* a tool was called |
| Strict separation | The subgraph never imports or calls MCP. KG reads are checked by Neo4j's planner (`EXPLAIN` query type must be read) inside READ sessions |
| New tools need no routing changes | `agent/mcp_tools.py` discovers tools from every server in `config.yaml` each time the graph is built |
| Correct MCP schemas | FastMCP generates JSON Schemas from typed signatures (`Literal` enums, `date` formats, bounds) plus `ToolAnnotations` (`readOnlyHint`, `destructiveHint`, …) and output schemas |

## Stack
LangGraph 1.x · langchain-anthropic (Claude Sonnet 5.5, adaptive thinking, `effort` medium/low,
server-side refusal fallbacks) · Qdrant + FastEmbed (local `bge-small-en-v1.5` + BM25, RRF fusion) ·
Neo4j 5 · MCP Python SDK (FastMCP, streamable HTTP) · langchain-mcp-adapters · LangChain Agent Chat UI.

## Run it

Requires Docker, Python 3.12+, Node 18+ (only for the UI), and an Anthropic API key.

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY
```

### Install dependencies (pick one)

**Option A: uv (recommended).** `uv sync` creates `.venv/` and installs the exact versions pinned in `uv.lock`:
```bash
uv sync
```
Prefix commands with `uv run` (as below), or activate `.venv` first.

**Option B: pip + venv.** `requirements.txt` and `requirements-dev.txt` are generated from `uv.lock` with the same pins:
```bash
python -m venv .venv
```
```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```
```bash
pip install -r requirements.txt -r requirements-dev.txt
```
With pip, drop the `uv run` prefix from the commands below, e.g. `python -m ingest.build_kg` or `langgraph dev`.
To regenerate the files after changing dependencies:
`uv export --no-hashes --no-dev --no-emit-project -o requirements.txt` and
`uv export --no-hashes --only-dev --no-emit-project -o requirements-dev.txt`.

### Start everything

```bash
docker compose up -d neo4j qdrant
```

```bash
uv run python -m ingest.build_kg          # CSV seed → Neo4j
```

```bash
QDRANT_URL=http://localhost:6333 uv run python -m ingest.build_vectors   # docs → Qdrant
```
(Leave out `QDRANT_URL` to use embedded Qdrant stored in `.qdrant/`. Use the same setting for the agent.)

Start the MCP server, either in Docker or locally:
```bash
docker compose up -d mcp-server
```
```bash
uv run python -m mcp_server.server
```

Start the agent server:
```bash
QDRANT_URL=http://localhost:6333 uv run langgraph dev
```

**UI:** open https://agentchat.vercel.app (or run `npx create-agent-chat-app`), set the deployment URL
to `http://localhost:2024` and the graph/assistant id to `agent`. For a terminal chat instead:
```bash
uv run python -m agent.chat
```

**Inspect the MCP server:**
```bash
npx @modelcontextprotocol/inspector
```
Connect to `http://localhost:8000/mcp` (streamable HTTP). `.vscode/mcp.json` makes the same server
available to VS Code / GitHub Copilot agent mode.

### Demo script
1. *"Which available electricians could start on Apollo next week, and do they meet Northwind's rules?"*
   → `knowledge_research` (graph candidates checked against MSA rules: OSHA 30, $125 cap)
2. *"Assign Ana Torres to Apollo as an electrician starting 2026-11-24."* → `assign_contractor`
3. *"Who's on Apollo now?"* → `knowledge_research` shows the new assignment from the KG
4. *"Put Chloe on the Northwind Clinic Expansion."* → MCP rejects it (project on hold), agent explains
5. *"Assign the best available plumber to Apollo from Monday."* → research first, then act

### Human approval (optional)
Set `REQUIRE_APPROVAL=true` (or `agent.require_approval_for_destructive` in `config.yaml`). MCP tools
whose annotation has `destructiveHint: true` then pause with an interrupt that Agent Chat UI
shows as accept / edit / ignore. The rule reads MCP metadata, not tool names.

## Extending
* **New action:** add a function in `mcp_server/tools/` and register it in `mcp_server/server.py`.
  Restart the MCP server. The agent picks it up with no change to `agent/`.
* **New MCP server:** add it under `mcp_servers:` in `config.yaml`.
* **New knowledge:** drop markdown files (with front matter) in `data/docs/` and re-run
  `build_vectors`; add CSV rows / statements in `ingest/build_kg.py`.

### Upgrading from Qdrant 1.12
`docker-compose.yml` now uses Qdrant 1.20 to match the Python client. If you started the stack
with the older image, recreate the Qdrant volume and re-index:
```bash
docker compose down
```
```bash
docker volume rm practical_challenge_qdrant_data
```
```bash
docker compose up -d
```
```bash
python -m ingest.build_vectors
```

## Tests
```bash
uv run pytest -q
```
* Offline (Neo4j + MCP server running, no API key needed): MCP schemas and annotations, business rules,
  audit log, read-only KG enforcement, RAG retrieval quality, subgraph shape, tool-call wiring.
* Live (needs `ANTHROPIC_API_KEY`): `tests/eval_set.jsonl` contains 20 labeled prompts and checks that
  Claude picks `knowledge_research` vs. the right MCP tool. A further test starts an *extra* MCP server
  with a new `send_reminder` tool and checks that the agent uses it with no code changes.

## Layout
```
agent/            supervisor graph, prompts, LLM factory, MCP tool loader, terminal chat
agent/knowledge/  RAG (rag.py), KG text→Cypher (kg.py), centralized subgraph + tool (subgraph.py)
mcp_server/       FastMCP server, tools/, db.py (write driver + JSONL audit log), Dockerfile
ingest/           build_kg.py (CSV → Neo4j), build_vectors.py (docs → Qdrant)
data/             csv/ seed data, docs/ contracts, SOWs, policies
tests/            pytest suites + eval_set.jsonl
```
