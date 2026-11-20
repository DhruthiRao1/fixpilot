"""Google ADK front-end.

The conversational shell is an ADK agent (so it can run in `adk web`, be deployed with
`adk deploy agent_engine`, and be composed with other ADK agents via A2A). The safety-
critical workflow lives in the LangGraph state machine, exposed to ADK as two tools.

Run locally:   cd src/fixpilot && adk web   (pick "adk_agent")
Deploy:        see deploy/README.md
"""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.tools import ToolContext

from fixpilot.graph import FixPilot

_pilot = FixPilot()
_SESSION_KEY = "fixpilot_session_id"


def _latest_image(tool_context: ToolContext) -> tuple[bytes | None, str]:
    content = tool_context.user_content
    for part in (content.parts if content and content.parts else []):
        blob = getattr(part, "inline_data", None)
        if blob and blob.data and (blob.mime_type or "").startswith("image/"):
            return blob.data, blob.mime_type
    return None, "image/jpeg"


def _render(result: dict) -> dict:
    """Return only what the shell model needs to relay to the user."""
    if result["status"] == "awaiting_user":
        p = result["prompt"]
        return {"status": "awaiting_user", "say_to_user": p["message"],
                "options": [o["label"] for o in p.get("options", [])], "hint": p.get("hint")}
    return {"status": result["status"], "say_to_user": result.get("message"),
            "ticket_id": result.get("ticket_id")}


def start_troubleshooting(user_message: str, tool_context: ToolContext) -> dict:
    """Start a new troubleshooting session from the user's message and the photo of their
    screen attached to this turn.

    Args:
        user_message: What the user said, verbatim.
    """
    image, mime = _latest_image(tool_context)
    result = _pilot.start(message=user_message, image=image, mime_type=mime)
    tool_context.state[_SESSION_KEY] = result["session_id"]
    return _render(result)


def continue_troubleshooting(user_reply: str, chosen_option: str, tool_context: ToolContext) -> dict:
    """Send the user's answer (and any new photo attached to this turn) to the active session.

    Args:
        user_reply: What the user said, verbatim.
        chosen_option: The option label the user picked, or an empty string.
    """
    sid = tool_context.state.get(_SESSION_KEY)
    if not sid:
        return {"status": "no_session", "say_to_user": "Please send a photo of your screen."}
    from fixpilot.graph.build import INTENT_OPTIONS

    choice = next((o["id"] for o in INTENT_OPTIONS if o["label"] == chosen_option), None)
    image, mime = _latest_image(tool_context)
    result = _pilot.reply(sid, text=user_reply, choice=choice, image=image, mime_type=mime)
    if result["status"] != "awaiting_user":
        tool_context.state[_SESSION_KEY] = None
    return _render(result)


root_agent = Agent(
    name="fixpilot",
    model=os.environ.get("FIXPILOT_MODEL", "gemini-2.5-flash"),
    description="Helps employees fix device problems from a photo of their screen.",
    instruction=(
        "You are FixPilot, an IT support assistant. When the user sends a photo of a screen "
        "or describes a device problem and no session is active, call start_troubleshooting. "
        "For every later reply, call continue_troubleshooting. Relay `say_to_user` to the user "
        "faithfully and in plain words; if `options` are present, list them. Never invent "
        "troubleshooting steps yourself and never tell the user to change BIOS, security or "
        "reset settings unless the tool's message says so."
    ),
    tools=[start_troubleshooting, continue_troubleshooting],
)
