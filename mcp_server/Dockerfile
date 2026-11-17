FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "mcp[cli]>=1.30" "neo4j>=6"
COPY mcp_server ./mcp_server
CMD ["python", "-m", "mcp_server.server"]
