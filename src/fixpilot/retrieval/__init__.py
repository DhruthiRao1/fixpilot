"""Retrieval over vendor manuals, internal KB articles and past tickets."""

from __future__ import annotations

from typing import Protocol

from ..config import Settings, get_settings
from ..schemas import RetrievedDoc


class Retriever(Protocol):
    def search(self, query: str, *, vendor: str | None = None, k: int = 4) -> list[RetrievedDoc]: ...


def build_retriever(settings: Settings | None = None) -> Retriever:
    settings = settings or get_settings()
    if settings.retriever == "local":
        from .local import LocalKeywordRetriever

        return LocalKeywordRetriever(settings.kb_dir)
    if settings.retriever == "vertex_rag":
        from .vertex_rag import VertexRagRetriever

        return VertexRagRetriever(settings)
    raise ValueError(f"Unknown FIXPILOT_RETRIEVER={settings.retriever!r}")
