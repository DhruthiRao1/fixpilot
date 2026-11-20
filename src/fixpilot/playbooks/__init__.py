"""Playbook library: vetted, risk-rated fix procedures."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..risk import annotate
from ..schemas import Intent, Plan, ScreenAnalysis, ScreenType, Step

_PATH = Path(__file__).with_name("playbooks.yaml")


class Match(BaseModel):
    intents: list[Intent]
    screens: list[ScreenType] = Field(default_factory=list)
    vendors: list[str] = Field(default_factory=list)


class Playbook(BaseModel):
    id: str
    title: str
    summary: str
    match: Match
    steps: list[Step]

    def matches(self, intent: Intent, screen: ScreenAnalysis) -> bool:
        if intent not in self.match.intents:
            return False
        if self.match.screens and screen.screen_type not in self.match.screens:
            return False
        if self.match.vendors:
            vendor = (screen.vendor or "").strip().lower()
            if vendor not in {v.lower() for v in self.match.vendors}:
                return False
        return True

    def to_plan(self, sources: list[str] | None = None) -> Plan:
        return Plan(
            playbook_id=self.id,
            title=self.title,
            summary=" ".join(self.summary.split()),
            steps=[annotate(s.model_copy(update={"instruction": " ".join(s.instruction.split())}))
                   for s in self.steps],
            sources=sources or [],
        )


@lru_cache(maxsize=1)
def load_playbooks() -> tuple[Playbook, ...]:
    raw = yaml.safe_load(_PATH.read_text())
    return tuple(Playbook.model_validate(p) for p in raw)


def candidates(intent: Intent, screen: ScreenAnalysis) -> list[Playbook]:
    return [p for p in load_playbooks() if p.matches(intent, screen)]


def get(playbook_id: str) -> Playbook | None:
    return next((p for p in load_playbooks() if p.id == playbook_id), None)
