# castor-server Dockerfile.
#
# Build context: the PARENT directory of castor-server (so the build can
# also see ../roche, which provides the roche-sandbox local editable
# dependency).
#
#   docker build -f castor-server/Dockerfile -t castor/server:latest ..
#
# Or via the bundled docker-compose.yml (which sets the right context):
#
#   docker compose up -d
#
# At runtime the container exposes port 8080 and stores its SQLite
# database in /data so the volume can be persisted across restarts.

FROM python:3.12-slim AS base

# System packages: git for any future repo cloning, curl for healthchecks,
# build essentials for any wheels that don't ship binaries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Bring in both source trees. The build context must be the parent dir
# of castor-server for this to resolve.
COPY castor-server /app/castor-server
COPY roche /app/roche

# Sync dependencies inside castor-server. uv reads pyproject.toml and
# the [tool.uv.sources] entry that points roche-sandbox at ../roche/sdk/python.
WORKDIR /app/castor-server
RUN uv sync --frozen || uv sync

# Persisted state lives here. CASTOR_DATABASE_URL points at this file by
# default in the compose file; override via env if needed.
RUN mkdir -p /data

ENV CASTOR_HOST=0.0.0.0 \
    CASTOR_PORT=8080 \
    CASTOR_DATABASE_URL="sqlite+aiosqlite:////data/castor.db"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uv", "run", "castor-server", "run"]
