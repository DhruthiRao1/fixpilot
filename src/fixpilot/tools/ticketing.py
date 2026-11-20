"""Escalation to a human: create an incident in the ITSM tool.

The ticket carries the redacted screen analysis, the conversation summary and the
reason the agent stopped, so the technician does not start from zero.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from ..config import Settings, get_settings
from ..pii import redact

log = logging.getLogger(__name__)


@dataclass
class Ticket:
    id: str
    short_description: str
    description: str
    priority: int
    url: str | None = None


class TicketClient(Protocol):
    def create(self, short_description: str, description: str, priority: int = 3) -> Ticket: ...


@dataclass
class MockTicketClient:
    tickets: list[Ticket] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1001))

    def create(self, short_description: str, description: str, priority: int = 3) -> Ticket:
        t = Ticket(f"INC{next(self._ids)}", redact(short_description), redact(description), priority)
        self.tickets.append(t)
        log.info("ticket_created id=%s priority=%s", t.id, priority)
        return t


class ServiceNowClient:
    """Minimal ServiceNow Table API client (incident table)."""

    def __init__(self, settings: Settings):
        if not (settings.servicenow_instance and settings.servicenow_user):
            raise RuntimeError("ServiceNow credentials are not configured")
        self._base = settings.servicenow_instance.rstrip("/")
        self._auth = (settings.servicenow_user, settings.servicenow_password or "")

    def create(self, short_description: str, description: str, priority: int = 3) -> Ticket:
        payload = {
            "short_description": redact(short_description)[:160],
            "description": redact(description),
            "urgency": str(priority),
            "impact": str(priority),
            "category": "hardware",
            "contact_type": "virtual_agent",
        }
        resp = httpx.post(
            f"{self._base}/api/now/table/incident",
            json=payload,
            auth=self._auth,
            headers={"Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        rec = resp.json()["result"]
        return Ticket(rec["number"], payload["short_description"], payload["description"],
                      priority, f"{self._base}/nav_to.do?uri=incident.do?sys_id={rec['sys_id']}")


def build_ticket_client(settings: Settings | None = None) -> TicketClient:
    settings = settings or get_settings()
    if settings.ticketing == "servicenow":
        return ServiceNowClient(settings)
    return MockTicketClient()
