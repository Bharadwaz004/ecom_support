# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Dependencies first: editing application code must not invalidate this layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only what the container actually runs. scripts/, evals/ and tests/ stay out - they carry
# torch and sentence-transformers, which would not fit the memory budget anyway.
COPY app/ ./app/
COPY static/ ./static/
COPY data/orders.db ./data/orders.db

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 200 whether healthy or degraded: a Qdrant outage should not restart a container that is
# still serving order lookups and the trace stream.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health', timeout=4)"

# MCP_SERVER_URL has to follow PORT, because the agent's MCP client dials this same
# process. Platforms that inject their own PORT would otherwise leave it pointing nowhere.
CMD ["sh", "-c", "export MCP_SERVER_URL=${MCP_SERVER_URL:-http://127.0.0.1:${PORT:-8000}/mcp/}; exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
