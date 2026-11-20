"""Offline BM25 retriever over Markdown files with YAML front matter.

Front matter fields (``vendor``, ``topic``) become metadata filters, mirroring the
metadata filtering used with the Vertex AI RAG Engine in production.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import yaml

from ..schemas import RetrievedDoc

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "to", "and", "or", "of", "in", "on", "is", "it", "you", "your",
         "for", "this", "that", "with", "be", "if", "then", "i", "my", "what", "how"}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _parse(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        return yaml.safe_load(fm) or {}, body.strip()
    return {}, raw


class LocalKeywordRetriever:
    def __init__(self, kb_dir: Path, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[tuple[str, dict, str, list[str]]] = []
        for path in sorted(Path(kb_dir).glob("*.md")):
            meta, body = _parse(path)
            self.docs.append((path.stem, meta, body, _tokens(body + " " + meta.get("title", ""))))
        self.avgdl = sum(len(d[3]) for d in self.docs) / max(len(self.docs), 1)
        df: Counter[str] = Counter()
        for *_, toks in self.docs:
            df.update(set(toks))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _score(self, q: list[str], toks: list[str]) -> float:
        tf = Counter(toks)
        score = 0.0
        for term in q:
            if term not in tf:
                continue
            f = tf[term]
            denom = f + self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl)
            score += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
        return score

    def search(self, query: str, *, vendor: str | None = None, k: int = 4) -> list[RetrievedDoc]:
        q = _tokens(query)
        results: list[RetrievedDoc] = []
        for doc_id, meta, body, toks in self.docs:
            doc_vendor = str(meta.get("vendor", "any")).lower()
            if vendor and doc_vendor not in {"any", vendor.lower()}:
                continue
            s = self._score(q, toks)
            if s > 0:
                results.append(RetrievedDoc(
                    doc_id=doc_id, title=meta.get("title", doc_id), text=body, score=round(s, 4),
                    metadata={k_: str(v) for k_, v in meta.items()},
                ))
        results.sort(key=lambda d: d.score, reverse=True)
        return results[:k]
