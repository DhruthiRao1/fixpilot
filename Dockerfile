FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY kb ./kb
RUN pip install ".[vertex-rag]"

# run as non-root
RUN useradd --create-home --uid 10001 app
USER app

ENV FIXPILOT_KB_DIR=/app/kb/docs PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn fixpilot.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
