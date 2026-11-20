"""Vertex AI RAG Engine retriever (production).

Ingest kb/docs (plus vendor PDFs and resolved tickets) into a RAG corpus with
``scripts/ingest_vertex_rag.py``, then set FIXPILOT_RETRIEVER=vertex_rag and
FIXPILOT_RAG_CORPUS=projects/.../ragCorpora/ID.
"""

from __future__ import annotations

from ..config import Settings
from ..schemas import RetrievedDoc


class VertexRagRetriever:
    def __init__(self, settings: Settings):
        if not settings.rag_corpus:
            raise RuntimeError("FIXPILOT_RAG_CORPUS must be set for the vertex_rag retriever")
        import vertexai
        from vertexai import rag

        vertexai.init(project=settings.gcp_project, location=settings.gcp_location)
        self._rag = rag
        self._corpus = settings.rag_corpus

    def search(self, query: str, *, vendor: str | None = None, k: int = 4) -> list[RetrievedDoc]:
        rag = self._rag
        # Vendor is folded into the query; with Vector Search backends, use a metadata
        # filter on `vendor` instead.
        text = f"{vendor} {query}" if vendor else query
        resp = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=self._corpus)],
            text=text,
            rag_retrieval_config=rag.RagRetrievalConfig(top_k=k),
        )
        docs: list[RetrievedDoc] = []
        for i, ctx in enumerate(resp.contexts.contexts):
            source = getattr(ctx, "source_uri", "") or f"ctx-{i}"
            score = getattr(ctx, "score", None)
            if score is None:
                score = 1.0 - float(getattr(ctx, "distance", 0.0) or 0.0)
            docs.append(RetrievedDoc(
                doc_id=source.rsplit("/", 1)[-1], title=source, text=ctx.text,
                score=float(score), metadata={"source_uri": source},
            ))
        return docs
