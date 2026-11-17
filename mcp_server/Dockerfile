FROM python:3.12-slim
WORKDIR /app
# Install only the MCP server's packages, constrained to the versions locked in uv.lock
# (requirements.txt is generated from it), so a new major release can't break the image.
COPY requirements.txt .
RUN pip install --no-cache-dir -c requirements.txt "mcp[cli]" neo4j
COPY mcp_server ./mcp_server
CMD ["python", "-m", "mcp_server.server"]
