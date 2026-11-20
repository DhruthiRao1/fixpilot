"""End-to-end behaviour of the LangGraph workflow (offline, MockLLM)."""

import pytest

from fixpilot.graph import FixPilot
from fixpilot.llm import MockLLM


def test_asks_intent_before_suggesting_anything(pilot, screen):
    r = pilot.start(message="", image=screen("hp_boot_menu"), mime_type="application/json")
    assert r["status"] == "awaiting_user"
    assert r["prompt"]["type"] == "question"
    assert "what were you trying to do" in r["prompt"]["message"].lower()
    assert r["playbook"] is None  # no plan exists yet


def test_accidental_boot_menu_resolves_with_no_setting_changes(pilot, screen):
    r = pilot.start("I was cleaning it and switched it on", screen("hp_boot_menu"), "application/json")
    assert r["playbook"] == "boot_menu_continue_normally"
    assert "Enter" in r["prompt"]["message"]
    r = pilot.reply(r["session_id"], text="done")
    r = pilot.reply(r["session_id"], text="done")
    assert r["status"] == "resolved"


def test_usb_flow_requires_recovery_key_then_explicit_confirmation(pilot, screen):
    r = pilot.start("install ubuntu from usb", screen("hp_boot_menu"), "application/json")
    assert r["prompt"]["type"] == "confirm" and "BitLocker" in r["prompt"]["message"]
    r = pilot.reply(r["session_id"], text="yes")
    assert r["prompt"]["type"] == "confirm" and "security setting" in r["prompt"]["message"]
    assert "to undo" in r["prompt"]["message"]
    r = pilot.reply(r["session_id"], text="yes")
    assert r["prompt"]["type"] == "step"


def test_declining_risk_changes_nothing(pilot, screen):
    r = pilot.start("boot from usb", screen("hp_boot_menu"), "application/json")
    r = pilot.reply(r["session_id"], text="yes")
    r = pilot.reply(r["session_id"], text="no")
    assert r["status"] == "declined"
    assert "no changes" in r["message"].lower()


def test_escalation_creates_redacted_ticket(pilot, tickets, screen):
    r = pilot.start("reset my laptop", screen("win_desktop"), "application/json")
    assert r["status"] == "escalated" and r["ticket_id"].startswith("INC")
    ticket = tickets.tickets[-1]
    assert "jane.doe@example.com" not in ticket.description


def test_raw_image_is_not_persisted(pilot, screen):
    r = pilot.start("", screen("hp_bios_main"), "application/json")
    state = pilot.graph.get_state({"configurable": {"thread_id": r["session_id"]}}).values
    assert state.get("pending_image_b64") is None
    assert "5CD1234XYZ" not in str(state)


def test_photo_verification_detects_wrong_screen(pilot, screen):
    r = pilot.start("switched it on, nothing I did", screen("hp_boot_menu"), "application/json")
    r = pilot.reply(r["session_id"], image=screen("hp_bios_main"), mime_type="application/json")
    assert r["prompt"]["type"] == "step"
    assert "not windows desktop" in r["prompt"]["message"].lower()


def test_unknown_session_rejected(pilot):
    with pytest.raises(KeyError):
        pilot.reply("does-not-exist", text="hi")


def test_low_confidence_photo_asks_for_new_photo(screen):
    p = FixPilot(llm=MockLLM())
    r = p.start("help", screen("blurry"), "application/json")
    assert r["prompt"]["type"] == "photo"
