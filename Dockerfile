# Multi-stage build: resolve/install dependencies with uv (using the committed
# lock file for a reproducible image), then copy the venv into a slim runtime
# stage so the final image doesn't carry uv or build tooling.

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Cache-friendly: dependencies change far less often than source, so resolve
# them from the lock file before copying the rest of the project. Both extras
# land in one venv, shared by both runtime targets below (providers = LLM
# SDKs for the API; ui = Streamlit for the frontend).
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --extra providers --extra ui

COPY src ./src
COPY README.md ./
RUN uv sync --locked --extra providers --extra ui


FROM python:3.13-slim AS runtime

# libgomp1: fastembed's ONNX runtime dependency at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 trustresume
WORKDIR /app

COPY --from=builder --chown=trustresume:trustresume /app/.venv ./.venv
COPY --chown=trustresume:trustresume src ./src
COPY --chown=trustresume:trustresume config ./config

# TRUSTRESUME_OUTPUT_DIR must be an absolute path under the volume. Its
# default is the relative "output", which resolves against WORKDIR /app —
# owned by root while the process runs as trustresume, so every generation
# would log a permission warning and write nothing. And even with permissions
# fixed, /app is the container's writable layer: résumés (real candidate data)
# would be destroyed by the next `docker compose up --build`.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    TRUSTRESUME_DB_PATH=/data/trustresume.db \
    TRUSTRESUME_CHROMA_PATH=/data/chroma_data \
    TRUSTRESUME_OUTPUT_DIR=/data/output

RUN mkdir -p /data && chown trustresume:trustresume /data
VOLUME ["/data"]

USER trustresume
EXPOSE 8000

CMD ["uvicorn", "trustresume.api.server:build_served_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]


# Same venv/source as "runtime", just a different entrypoint — the Streamlit
# frontend talks to the API stage over HTTP (TRUSTRESUME_API_URL), not to
# SQLite/Chroma directly, so it needs no volume of its own.
FROM runtime AS ui

EXPOSE 8501

CMD ["streamlit", "run", "src/trustresume/ui/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
