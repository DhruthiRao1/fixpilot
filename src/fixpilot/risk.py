"""Risk registry and gate.

Every step a playbook can emit references an *action id*. The registry is the single
source of truth for how risky that action is, whether it can be undone, and what must
be true before the agent is allowed to suggest it. The LLM never decides risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import RISK_ORDER, Intent, RiskLevel, Step


@dataclass(frozen=True)
class ActionPolicy:
    risk: RiskLevel
    reversible: bool
    # intents for which this action is appropriate; empty = any intent
    allowed_intents: frozenset[Intent] = field(default_factory=frozenset)
    # facts that must be confirmed by the user before this step is shown
    prerequisites: tuple[str, ...] = ()
    undo: str | None = None


REGISTRY: dict[str, ActionPolicy] = {
    "observe": ActionPolicy(RiskLevel.NONE, True),
    "select_default_os": ActionPolicy(RiskLevel.NONE, True),
    "check_stuck_keys": ActionPolicy(RiskLevel.NONE, True),
    "enter_bios": ActionPolicy(RiskLevel.LOW, True),
    "navigate_menu": ActionPolicy(RiskLevel.LOW, True),
    "open_boot_menu": ActionPolicy(RiskLevel.LOW, True),
    "advanced_startup_uefi": ActionPolicy(RiskLevel.LOW, True),
    "verify_usb_media": ActionPolicy(RiskLevel.NONE, True),
    "change_hotkey_delay": ActionPolicy(
        RiskLevel.MEDIUM, True, undo="Set POST HotKey Delay back to its previous value."
    ),
    "change_boot_order": ActionPolicy(
        RiskLevel.MEDIUM, True, undo="Move OS Boot Manager back to the top of UEFI Boot Order."
    ),
    "enable_usb_boot": ActionPolicy(
        RiskLevel.MEDIUM, True, frozenset({Intent.BOOT_FROM_USB}),
        undo="Set USB Boot back to Disabled.",
    ),
    "disable_secure_boot": ActionPolicy(
        RiskLevel.HIGH,
        True,
        frozenset({Intent.BOOT_FROM_USB}),
        prerequisites=("bitlocker_key_available",),
        undo="Boot Options -> Secure Boot -> Enabled, save, and enter the confirmation code.",
    ),
    "enable_secure_boot": ActionPolicy(
        RiskLevel.HIGH,
        True,
        frozenset({Intent.RESTORE_SETTINGS}),
        prerequisites=("bitlocker_key_available",),
        undo="Boot Options -> Secure Boot -> Disabled.",
    ),
    "save_and_exit": ActionPolicy(RiskLevel.MEDIUM, True),
    "confirm_secure_boot_code": ActionPolicy(RiskLevel.MEDIUM, True),
    "retrieve_recovery_key": ActionPolicy(RiskLevel.LOW, True),
    "clear_tpm": ActionPolicy(RiskLevel.CRITICAL, False),
    "reset_pc": ActionPolicy(RiskLevel.CRITICAL, False, frozenset({Intent.RESET_DEVICE})),
    "escalate": ActionPolicy(RiskLevel.NONE, True),
}


PREREQUISITE_QUESTIONS: dict[str, str] = {
    "bitlocker_key_available": (
        "Changing Secure Boot can trigger a BitLocker recovery screen on the next start. "
        "Do you have your BitLocker recovery key (or have you confirmed the drive is not "
        "encrypted)? Reply yes or no."
    ),
}


class UnknownActionError(ValueError):
    pass


def policy_for(action: str) -> ActionPolicy:
    try:
        return REGISTRY[action]
    except KeyError as exc:  # a playbook referenced an action nobody has risk-rated
        raise UnknownActionError(f"Action '{action}' is not in the risk registry") from exc


def annotate(step: Step) -> Step:
    """Overwrite risk/reversibility on a step with the registry's values."""
    pol = policy_for(step.action)
    return step.model_copy(update={"risk": pol.risk, "reversible": pol.reversible, "undo": pol.undo})


def max_risk(steps: list[Step]) -> RiskLevel:
    if not steps:
        return RiskLevel.NONE
    return max((s.risk for s in steps), key=RISK_ORDER.index)


@dataclass
class GateDecision:
    allowed: bool
    needs_confirmation: bool
    escalate: bool
    missing_prerequisites: list[str]
    reasons: list[str]


def evaluate(steps: list[Step], intent: Intent, confirmed_facts: set[str]) -> GateDecision:
    reasons: list[str] = []
    missing: list[str] = []
    escalate = False
    needs_confirmation = False

    for step in steps:
        pol = policy_for(step.action)
        if pol.allowed_intents and intent not in pol.allowed_intents:
            reasons.append(
                f"Step '{step.id}' ({step.action}) is not appropriate for intent '{intent.value}'."
            )
            escalate = True
        if pol.risk == RiskLevel.CRITICAL:
            reasons.append(f"Step '{step.id}' ({step.action}) can cause data loss; human only.")
            escalate = True
        if pol.risk == RiskLevel.HIGH:
            needs_confirmation = True
        for prereq in pol.prerequisites:
            if prereq not in confirmed_facts and prereq not in missing:
                missing.append(prereq)

    return GateDecision(
        allowed=not escalate,
        needs_confirmation=needs_confirmation,
        escalate=escalate,
        missing_prerequisites=missing,
        reasons=reasons,
    )
