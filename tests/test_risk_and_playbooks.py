import pytest

from fixpilot import playbooks as pb
from fixpilot.risk import REGISTRY, UnknownActionError, evaluate, policy_for
from fixpilot.schemas import Intent, RiskLevel, Step


def test_every_playbook_action_is_risk_rated():
    for book in pb.load_playbooks():
        for step in book.steps:
            assert step.action in REGISTRY, f"{book.id}:{step.id} uses unrated action {step.action}"


def test_unknown_action_raises():
    with pytest.raises(UnknownActionError):
        policy_for("format_disk")


def test_registry_overrides_yaml_risk():
    plan = pb.get("hp_usb_boot").to_plan()
    secure = next(s for s in plan.steps if s.action == "disable_secure_boot")
    assert secure.risk == RiskLevel.HIGH
    assert secure.undo


def test_disabling_secure_boot_is_blocked_for_wrong_intent():
    step = Step(id="x", instruction="...", action="disable_secure_boot")
    decision = evaluate([step], Intent.START_NORMALLY, confirmed_facts={"bitlocker_key_available"})
    assert decision.escalate and not decision.allowed


def test_high_risk_needs_prereq_and_confirmation():
    step = Step(id="x", instruction="...", action="disable_secure_boot")
    d = evaluate([step], Intent.BOOT_FROM_USB, confirmed_facts=set())
    assert d.missing_prerequisites == ["bitlocker_key_available"]
    assert d.needs_confirmation and d.allowed


def test_critical_actions_always_escalate():
    step = Step(id="x", instruction="...", action="reset_pc")
    assert evaluate([step], Intent.RESET_DEVICE, set()).escalate


def test_accidental_boot_menu_playbook_changes_nothing():
    plan = pb.get("boot_menu_continue_normally").to_plan()
    assert all(s.risk in (RiskLevel.NONE, RiskLevel.LOW) for s in plan.steps)
