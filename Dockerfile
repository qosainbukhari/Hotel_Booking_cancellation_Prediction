# Serving image: API + web UI + the production model's serving bundle.
# Build after training (it copies artifacts/serving):  uv run hc-train && docker build -t radar .

# ---- Stage 1: build the virtualenv with runtime dependencies only -----------------------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
# Dependencies first so code changes don't reinstall them. No dev/train groups.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project --no-editable
COPY src ./src
RUN uv sync --frozen --no-default-groups --no-editable

# ---- Stage 2: slim runtime, no uv, no source tree, no build caches ----------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app /app/.venv ./.venv
COPY --chown=app api ./api
COPY --chown=app frontend ./frontend
COPY --chown=app configs ./configs
# The model ships inside the image, so the container runs anywhere (e.g. Hugging Face Spaces).
COPY --chown=app artifacts/serving ./artifacts/serving
RUN mkdir -p logs && chown app logs

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
