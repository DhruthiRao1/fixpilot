"""Typed contracts shared by every component.

Keeping the LLM's outputs inside strict Pydantic schemas is what lets the rest of the
system (risk gate, playbook validator, evals) reason about them deterministically.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ScreenType(str, Enum):
    BOOT_MENU = "boot_menu"                  # "Please select boot device"
    BIOS_SETUP = "bios_setup"                # BIOS Setup Utility (any tab)
    SECURE_BOOT_CONFIRM = "secure_boot_confirm"  # HP "Operating System Boot Mode Change" code
    BITLOCKER_RECOVERY = "bitlocker_recovery"
    WINDOWS_RECOVERY = "windows_recovery"    # "Choose an option" (WinRE)
    WINDOWS_DESKTOP = "windows_desktop"
    BLUE_SCREEN = "blue_screen"              # BSOD / stop code
    BLACK_SCREEN = "black_screen"
    UNKNOWN = "unknown"


class Intent(str, Enum):
    START_NORMALLY = "start_normally"        # "I just want my laptop to work"
    BOOT_FROM_USB = "boot_from_usb"          # install / repair from USB
    RESTORE_SETTINGS = "restore_settings"    # undo earlier BIOS changes
    FIX_ERROR = "fix_error"
    RESET_DEVICE = "reset_device"            # wipe / reinstall
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    NONE = "none"          # reading, pressing Enter on the default
    LOW = "low"            # navigating menus, reversible, no security impact
    MEDIUM = "medium"      # persistent setting change, easily reversible
    HIGH = "high"          # security-relevant or may lock the user out (Secure Boot, TPM)
    CRITICAL = "critical"  # data loss possible (reset, format, clear TPM) -> human only


RISK_ORDER = [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]


class ScreenAnalysis(BaseModel):
    """What the vision model saw in the user's photo."""

    screen_type: ScreenType = ScreenType.UNKNOWN
    vendor: str | None = Field(None, description="e.g. HP, Dell, Lenovo")
    model: str | None = Field(None, description="e.g. HP Laptop 15-dy2xxx")
    visible_text: str = Field("", description="Key on-screen text, PII redacted")
    settings: dict[str, str] = Field(default_factory=dict, description="Visible setting -> value")
    options: list[str] = Field(default_factory=list, description="Selectable options on screen")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class IntentResult(BaseModel):
    intent: Intent = Intent.UNKNOWN
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str = ""


class Step(BaseModel):
    id: str
    instruction: str
    action: str = Field(..., description="Action id from the risk registry")
    expected_screen: ScreenType | None = None
    risk: RiskLevel = RiskLevel.NONE
    reversible: bool = True
    undo: str | None = None


class Plan(BaseModel):
    playbook_id: str
    title: str
    summary: str
    steps: list[Step]
    sources: list[str] = Field(default_factory=list)


class RetrievedDoc(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)
