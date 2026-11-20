"""Graph state. Everything here is persisted by the checkpointer, so raw images are
dropped as soon as they have been analysed and all text is PII-redacted."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # inputs
    user_message: str
    pending_image_b64: str | None
    pending_mime_type: str | None

    # perception
    screen: dict[str, Any]
    photo_requests: int

    # intent
    intent: str
    intent_rationale: str
    intent_questions: int
    conversation: Annotated[list[str], operator.add]

    # knowledge + plan
    docs: list[dict[str, Any]]
    plan: dict[str, Any] | None
    plan_explanation: str

    # guardrails
    confirmed_facts: list[str]
    prereq_declines: int
    risk_confirmed: bool
    gate: dict[str, Any]

    # execution
    step_index: int
    step_note: str
    last_reply: dict[str, Any]
    mismatches: int

    # outcome
    status: str                      # in_progress | resolved | declined | escalated
    outcome_message: str
    escalation_reason: str
    ticket_id: str | None
    trace: Annotated[list[dict[str, Any]], operator.add]
