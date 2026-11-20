"""Runtime configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    llm_backend: str = "mock"
    gcp_project: str | None = None
    gcp_location: str = "us-central1"
    model: str = "gemini-2.5-flash"
    retriever: str = "local"
    rag_corpus: str | None = None
    ticketing: str = "mock"
    servicenow_instance: str | None = None
    servicenow_user: str | None = None
    servicenow_password: str | None = None
    min_vision_confidence: float = 0.6
    kb_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "kb" / "docs")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        llm_backend=env("FIXPILOT_LLM_BACKEND", "mock").lower(),
        gcp_project=env("GOOGLE_CLOUD_PROJECT"),
        gcp_location=env("GOOGLE_CLOUD_LOCATION", "us-central1"),
        model=env("FIXPILOT_MODEL", "gemini-2.5-flash"),
        retriever=env("FIXPILOT_RETRIEVER", "local").lower(),
        rag_corpus=env("FIXPILOT_RAG_CORPUS"),
        ticketing=env("FIXPILOT_TICKETING", "mock").lower(),
        servicenow_instance=env("SERVICENOW_INSTANCE"),
        servicenow_user=env("SERVICENOW_USER"),
        servicenow_password=env("SERVICENOW_PASSWORD"),
        min_vision_confidence=float(env("FIXPILOT_MIN_VISION_CONFIDENCE", "0.6")),
        kb_dir=Path(env("FIXPILOT_KB_DIR", str(PROJECT_ROOT / "kb" / "docs"))),
    )
