"""The FixPilot LangGraph workflow.

    analyze_screen ──(low confidence)──> ask_photo ──┐
          │                                          │
          ▼                                          │
    classify_intent ──(unknown)──> ask_intent ──┐    │
          │                                     │    │
          ▼                                     ▼    ▼
       retrieve ─> plan ─> risk_gate ─┬─> ask_prerequisite ─┐
                                      ├─> confirm_risk ─────┤
                                      ├─> escalate          │
                                      ▼                     │
                     ┌──────────> guide_step <──────────────┘
                     │                │ (user reply / next photo)
                     │                ▼
                     └────────── verify_step ──> finish | escalate

Design rules
* Each node performs at most one ``interrupt()`` and interrupting nodes have no side
  effects before the interrupt (LangGraph re-runs a node from the top on resume).
* The model reads screens, classifies intent and picks among *vetted* playbooks. Risk,
  prerequisites and escalation are decided by code (``fixpilot.risk``), never the model.
* The agent never assumes intent from the screen alone; it asks.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .. import playbooks as pb
from ..config import Settings, get_settings
from ..llm import LLM, build_llm
from ..pii import redact, redact_settings
from ..retrieval import Retriever, build_retriever
from ..risk import PREREQUISITE_QUESTIONS, evaluate
from ..schemas import Intent, Plan, RiskLevel, ScreenAnalysis, ScreenType
from ..tools.ticketing import TicketClient, build_ticket_client
from .state import AgentState

log = logging.getLogger(__name__)

MAX_PHOTO_REQUESTS = 2
MAX_INTENT_QUESTIONS = 2
MAX_PREREQ_DECLINES = 2
MAX_MISMATCHES = 2

INTENT_OPTIONS = [
    {"id": Intent.START_NORMALLY.value,
     "label": "Nothing - I just switched it on and want Windows to start normally"},
    {"id": Intent.BOOT_FROM_USB.value, "label": "Start from a USB drive (install or repair)"},
    {"id": Intent.RESTORE_SETTINGS.value, "label": "Undo settings I changed earlier"},
    {"id": Intent.FIX_ERROR.value, "label": "Fix an error message"},
    {"id": Intent.RESET_DEVICE.value, "label": "Reset or reinstall Windows"},
]

_YES = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "done", "next", "confirm", "haan", "ha"}
_NO = {"no", "n", "nope", "nah", "cancel", "stop"}


# ----------------------------------------------------------------------------- helpers


def _norm_reply(value: Any) -> dict[str, Any]:
    """Resume values may be a plain string or {"text", "choice", "image_b64", "mime_type"}."""
    if isinstance(value, str):
        return {"text": value}
    return dict(value or {})


def _yes_no(text: str | None) -> bool | None:
    words = set((text or "").lower().replace(",", " ").replace(".", " ").split())
    if words & _NO:
        return False
    if words & _YES:
        return True
    return None


def _trace(node: str, **data: Any) -> list[dict[str, Any]]:
    return [{"node": node, "ts": round(time.time(), 3), **data}]


@dataclass
class Deps:
    llm: LLM
    retriever: Retriever
    tickets: TicketClient
    settings: Settings


# ----------------------------------------------------------------------------- nodes


def make_nodes(d: Deps) -> dict[str, Any]:

    def analyze_screen(state: AgentState) -> dict:
        b64 = state.get("pending_image_b64")
        if not b64:
            screen = ScreenAnalysis()
        else:
            started = time.perf_counter()
            screen = d.llm.analyze_screen(base64.b64decode(b64),
                                          state.get("pending_mime_type") or "image/jpeg")
            log.info("vision screen=%s conf=%.2f latency_ms=%d", screen.screen_type.value,
                     screen.confidence, (time.perf_counter() - started) * 1000)
        screen = screen.model_copy(update={
            "visible_text": redact(screen.visible_text),
            "settings": redact_settings(screen.settings),
        })
        return {
            "screen": screen.model_dump(mode="json"),
            "pending_image_b64": None,  # never persist the raw photo
            "pending_mime_type": None,
            "trace": _trace("analyze_screen", screen_type=screen.screen_type.value,
                            confidence=screen.confidence),
        }

    def route_after_analyze(state: AgentState) -> str:
        screen = ScreenAnalysis.model_validate(state["screen"])
        ok = (screen.screen_type != ScreenType.UNKNOWN
              and screen.confidence >= d.settings.min_vision_confidence)
        if ok:
            return "classify_intent"
        if state.get("photo_requests", 0) >= MAX_PHOTO_REQUESTS:
            return "escalate"
        return "ask_photo"

    def ask_photo(state: AgentState) -> dict:
        reply = _norm_reply(interrupt({
            "type": "photo",
            "message": ("I can't read that screen clearly enough to help safely. Please send "
                        "a sharper photo of the whole screen, straight on, without glare."),
        }))
        return {
            "pending_image_b64": reply.get("image_b64"),
            "pending_mime_type": reply.get("mime_type"),
            "photo_requests": state.get("photo_requests", 0) + 1,
            "escalation_reason": "Screen could not be recognised from the photos provided.",
        }

    def classify_intent(state: AgentState) -> dict:
        if state.get("intent") and state["intent"] != Intent.UNKNOWN.value:
            return {}  # already set explicitly by the user choosing an option
        screen = ScreenAnalysis.model_validate(state["screen"])
        text = " ".join(state.get("conversation", []))
        result = d.llm.classify_intent(text, screen)
        return {
            "intent": result.intent.value,
            "intent_rationale": result.rationale,
            "trace": _trace("classify_intent", intent=result.intent.value,
                            confidence=result.confidence),
        }

    def route_after_intent(state: AgentState) -> str:
        if state.get("intent", Intent.UNKNOWN.value) != Intent.UNKNOWN.value:
            return "retrieve"
        if state.get("intent_questions", 0) >= MAX_INTENT_QUESTIONS:
            return "escalate"
        return "ask_intent"

    def ask_intent(state: AgentState) -> dict:
        screen = ScreenAnalysis.model_validate(state["screen"])
        label = screen.screen_type.value.replace("_", " ")
        reply = _norm_reply(interrupt({
            "type": "question",
            "message": (f"I can see a {label} screen"
                        f"{' on your ' + screen.model if screen.model else ''}. Before I suggest "
                        "any changes: what were you trying to do when this appeared?"),
            "options": INTENT_OPTIONS,
        }))
        update: dict[str, Any] = {"intent_questions": state.get("intent_questions", 0) + 1,
                                  "escalation_reason": "User's goal remained unclear."}
        choice = reply.get("choice")
        if choice in {i.value for i in Intent}:
            update["intent"] = choice
        if reply.get("text"):
            update["conversation"] = [redact(reply["text"])]
        return update

    def retrieve(state: AgentState) -> dict:
        screen = ScreenAnalysis.model_validate(state["screen"])
        query = " ".join(filter(None, [
            screen.screen_type.value.replace("_", " "), screen.model, screen.visible_text[:300],
            state["intent"].replace("_", " "), " ".join(state.get("conversation", []))[:300],
        ]))
        docs = d.retriever.search(query, vendor=screen.vendor, k=4)
        return {"docs": [doc.model_dump() for doc in docs],
                "trace": _trace("retrieve", doc_ids=[x.doc_id for x in docs])}

    def plan(state: AgentState) -> dict:
        from ..schemas import RetrievedDoc

        screen = ScreenAnalysis.model_validate(state["screen"])
        intent = Intent(state["intent"])
        cands = pb.candidates(intent, screen)
        docs = [RetrievedDoc.model_validate(x) for x in state.get("docs", [])]
        choice = d.llm.choose_playbook(
            intent, screen,
            [{"id": c.id, "title": c.title, "summary": " ".join(c.summary.split())} for c in cands],
            docs,
        )
        playbook = pb.get(choice.playbook_id) if choice.playbook_id else None
        if playbook is None:
            return {"plan": None,
                    "escalation_reason": (f"No vetted playbook for intent '{intent.value}' on a "
                                          f"'{screen.screen_type.value}' screen."),
                    "trace": _trace("plan", playbook=None)}
        p = playbook.to_plan(sources=[x.doc_id for x in docs])
        return {"plan": p.model_dump(mode="json"), "plan_explanation": choice.explanation,
                "step_index": 0, "mismatches": 0,
                "trace": _trace("plan", playbook=p.playbook_id, steps=len(p.steps))}

    def risk_gate(state: AgentState) -> dict:
        if not state.get("plan"):
            return {"gate": {"escalate": True}}
        p = Plan.model_validate(state["plan"])
        decision = evaluate(p.steps, Intent(state["intent"]), set(state.get("confirmed_facts", [])))
        update: dict[str, Any] = {"gate": decision.__dict__,
                                  "trace": _trace("risk_gate", **decision.__dict__)}
        if decision.escalate:
            update["escalation_reason"] = "; ".join(decision.reasons)
        return update

    def route_after_gate(state: AgentState) -> str:
        g = state["gate"]
        if g.get("escalate"):
            return "escalate"
        if g.get("missing_prerequisites"):
            return "ask_prerequisite"
        if g.get("needs_confirmation") and not state.get("risk_confirmed"):
            return "confirm_risk"
        return "guide_step"

    def ask_prerequisite(state: AgentState) -> dict:
        fact = state["gate"]["missing_prerequisites"][0]
        msg = PREREQUISITE_QUESTIONS.get(fact, f"Please confirm: {fact}")
        if state.get("prereq_declines"):
            msg = ("No problem - let's get that first. Check File Explorer -> This PC for a "
                   "padlock on C:, or get the key at aka.ms/myrecoverykey. " + msg)
        reply = _norm_reply(interrupt({"type": "confirm", "message": msg, "fact": fact}))
        answer = _yes_no(reply.get("text") or reply.get("choice"))
        if answer:
            return {"confirmed_facts": [*state.get("confirmed_facts", []), fact]}
        return {"prereq_declines": state.get("prereq_declines", 0) + 1,
                "escalation_reason": f"Prerequisite not met: {fact}"}

    def route_after_prereq(state: AgentState) -> str:
        if state.get("prereq_declines", 0) >= MAX_PREREQ_DECLINES:
            return "escalate"
        return "risk_gate"

    def confirm_risk(state: AgentState) -> dict:
        p = Plan.model_validate(state["plan"])
        risky = [s for s in p.steps if s.risk in (RiskLevel.HIGH,)]
        lines = [f"- {s.instruction} (to undo: {s.undo})" for s in risky]
        reply = _norm_reply(interrupt({
            "type": "confirm",
            "message": ("This fix changes a security setting:\n" + "\n".join(lines)
                        + "\nDo you want to go ahead? Reply yes or no."),
        }))
        if _yes_no(reply.get("text") or reply.get("choice")):
            return {"risk_confirmed": True}
        return {"risk_confirmed": False, "status": "declined",
                "outcome_message": "Okay - no changes made. Your settings are as they were."}

    def route_after_confirm(state: AgentState) -> str:
        return "guide_step" if state.get("risk_confirmed") else "finish"

    def guide_step(state: AgentState) -> dict:
        p = Plan.model_validate(state["plan"])
        i = state.get("step_index", 0)
        step = p.steps[i]
        header = f"{p.title} - step {i + 1} of {len(p.steps)}"
        intro = (state.get("plan_explanation", "") + "\n\n") if i == 0 else ""
        note = (state.get("step_note") or "")
        reply = _norm_reply(interrupt({
            "type": "step",
            "message": f"{note}{intro}{header}:\n{step.instruction}",
            "step_id": step.id,
            "risk": step.risk.value,
            "hint": "Reply 'done' when finished, send a photo of the new screen, or 'help'.",
        }))
        return {"last_reply": reply}

    def verify_step(state: AgentState) -> dict:
        p = Plan.model_validate(state["plan"])
        i = state.get("step_index", 0)
        step = p.steps[i]
        reply = state.get("last_reply", {})
        text = (reply.get("text") or "").lower()

        if any(w in text for w in ("help", "stuck", "human", "technician")):
            return {"status": "escalate",
                    "escalation_reason": f"User asked for help at step '{step.id}'."}

        if reply.get("image_b64"):
            screen = d.llm.analyze_screen(base64.b64decode(reply["image_b64"]),
                                          reply.get("mime_type") or "image/jpeg")
            if step.expected_screen and screen.screen_type != step.expected_screen:
                n = state.get("mismatches", 0) + 1
                return {"mismatches": n,
                        "step_note": (f"That looks like a {screen.screen_type.value.replace('_', ' ')}"
                                      f" screen, not {step.expected_screen.value.replace('_', ' ')}. "
                                      "Let's retry this step.\n\n"),
                        "escalation_reason": f"Screen did not match after step '{step.id}'.",
                        "trace": _trace("verify_step", step=step.id, match=False)}
        elif _yes_no(text) is False:
            return {"status": "declined",
                    "outcome_message": "Stopped at your request. Nothing further was changed."}

        nxt = i + 1
        update: dict[str, Any] = {"step_index": nxt, "step_note": "", "mismatches": 0,
                                  "trace": _trace("verify_step", step=step.id, match=True)}
        if nxt >= len(p.steps):
            update.update(status="resolved",
                          outcome_message="All steps done - you should be back to normal.")
        return update

    def route_after_verify(state: AgentState) -> str:
        if state.get("status") in {"resolved", "declined"}:
            return "finish"
        if state.get("status") == "escalate" or state.get("mismatches", 0) > MAX_MISMATCHES:
            return "escalate"
        return "guide_step"

    def escalate(state: AgentState) -> dict:
        screen = state.get("screen", {})
        reason = state.get("escalation_reason") or "Agent could not resolve safely."
        description = "\n".join([
            f"Reason: {reason}",
            (f"Screen: {screen.get('screen_type')} | vendor={screen.get('vendor')} "
             f"model={screen.get('model')}"),
            f"Visible text: {screen.get('visible_text', '')[:500]}",
            f"Intent: {state.get('intent')}",
            f"User said: {' | '.join(state.get('conversation', []))[:800]}",
            (f"Plan: {(state.get('plan') or {}).get('playbook_id')} "
             f"step={state.get('step_index')}"),
        ])
        ticket = d.tickets.create(f"FixPilot escalation: {reason[:100]}", description, priority=3)
        return {"status": "escalated", "ticket_id": ticket.id,
                "outcome_message": (f"I've handed this to a technician (ticket {ticket.id}) "
                                    "rather than guess. Please don't change anything else meanwhile."),
                "trace": _trace("escalate", ticket=ticket.id, reason=reason)}

    def finish(state: AgentState) -> dict:
        return {"status": state.get("status") or "resolved",
                "trace": _trace("finish", status=state.get("status"))}

    return {k: v for k, v in locals().items() if callable(v) and not k.startswith("_")}


# ----------------------------------------------------------------------------- graph


def build_graph(llm: LLM | None = None, retriever: Retriever | None = None,
                tickets: TicketClient | None = None, settings: Settings | None = None,
                checkpointer=None):
    settings = settings or get_settings()
    deps = Deps(llm or build_llm(settings), retriever or build_retriever(settings),
                tickets or build_ticket_client(settings), settings)
    n = make_nodes(deps)

    g = StateGraph(AgentState)
    for name in ["analyze_screen", "ask_photo", "classify_intent", "ask_intent", "retrieve",
                 "plan", "risk_gate", "ask_prerequisite", "confirm_risk", "guide_step",
                 "verify_step", "escalate", "finish"]:
        g.add_node(name, n[name])

    g.add_edge(START, "analyze_screen")
    g.add_conditional_edges("analyze_screen", n["route_after_analyze"],
                            ["classify_intent", "ask_photo", "escalate"])
    g.add_edge("ask_photo", "analyze_screen")
    g.add_conditional_edges("classify_intent", n["route_after_intent"],
                            ["retrieve", "ask_intent", "escalate"])
    g.add_edge("ask_intent", "classify_intent")
    g.add_edge("retrieve", "plan")
    g.add_edge("plan", "risk_gate")
    g.add_conditional_edges("risk_gate", n["route_after_gate"],
                            ["escalate", "ask_prerequisite", "confirm_risk", "guide_step"])
    g.add_conditional_edges("ask_prerequisite", n["route_after_prereq"], ["risk_gate", "escalate"])
    g.add_conditional_edges("confirm_risk", n["route_after_confirm"], ["guide_step", "finish"])
    g.add_edge("guide_step", "verify_step")
    g.add_conditional_edges("verify_step", n["route_after_verify"],
                            ["guide_step", "finish", "escalate"])
    g.add_edge("escalate", END)
    g.add_edge("finish", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver()), deps


# ----------------------------------------------------------------------------- facade


class FixPilot:
    """Session-oriented wrapper used by the API, the ADK agent and the evals."""

    def __init__(self, **kwargs):
        self.graph, self.deps = build_graph(**kwargs)

    def _result(self, session_id: str, out: dict) -> dict:
        pending = out.get("__interrupt__")
        state = self.graph.get_state({"configurable": {"thread_id": session_id}}).values
        base = {"session_id": session_id, "screen": state.get("screen"),
                "intent": state.get("intent"),
                "playbook": (state.get("plan") or {}).get("playbook_id")}
        if pending:
            return {**base, "status": "awaiting_user", "prompt": pending[0].value}
        return {**base, "status": state.get("status"), "message": state.get("outcome_message"),
                "ticket_id": state.get("ticket_id")}

    def start(self, message: str = "", image: bytes | None = None,
              mime_type: str = "image/jpeg", session_id: str | None = None) -> dict:
        session_id = session_id or uuid.uuid4().hex
        inputs: AgentState = {
            "user_message": redact(message),
            "conversation": [redact(message)] if message else [],
            "pending_image_b64": base64.b64encode(image).decode() if image else None,
            "pending_mime_type": mime_type,
            "status": "in_progress",
        }
        out = self.graph.invoke(inputs, {"configurable": {"thread_id": session_id}})
        return self._result(session_id, out)

    def reply(self, session_id: str, text: str | None = None, choice: str | None = None,
              image: bytes | None = None, mime_type: str = "image/jpeg") -> dict:
        config = {"configurable": {"thread_id": session_id}}
        if not self.graph.get_state(config).next:
            raise KeyError(f"No active session {session_id}")
        resume: dict[str, Any] = {"text": text, "choice": choice}
        if image:
            resume.update(image_b64=base64.b64encode(image).decode(), mime_type=mime_type)
        out = self.graph.invoke(Command(resume=resume), {"configurable": {"thread_id": session_id}})
        return self._result(session_id, out)

    def trace(self, session_id: str) -> list[dict]:
        return self.graph.get_state({"configurable": {"thread_id": session_id}}).values.get("trace", [])
