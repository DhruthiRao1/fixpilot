"""PII redaction.

Screenshots of BIOS / Settings pages routinely contain serial numbers, UUIDs, product
keys and the signed-in user's email. We redact before anything is logged, stored in
session state, sent to retrieval, or written to a ticket.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # UUID, including the line-wrapped form BIOS screens produce ("...-3953383 34435")
    ("UUID", re.compile(r"\b[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{6,12}(?:\s+[0-9A-Fa-f]{1,8})?\b")),
    ("PRODUCT_KEY", re.compile(r"\b(?:[A-Z0-9]{5}-){4}[A-Z0-9]{5}\b")),
    # BitLocker recovery password: 8 groups of 6 digits
    ("RECOVERY_KEY", re.compile(r"\b(?:\d{6}-){7}\d{6}\b")),
]

# Labelled fields: redact the value that follows the label on BIOS/system-info screens
_LABELLED = re.compile(
    r"(?P<label>serial number|system board ct number|primary battery sn|uuid|feature byte|build id)"
    r"(?P<sep>\s*[:=]?\s*)(?P<value>[^\n]+)",
    re.IGNORECASE,
)

SENSITIVE_SETTING_KEYS = {
    "serial number", "uuid", "system board ct number", "primary battery sn",
    "feature byte", "build id",
}


def redact(text: str) -> str:
    if not text:
        return text
    out = _LABELLED.sub(lambda m: f"{m['label']}{m['sep']}[REDACTED]", text)
    for name, pattern in _PATTERNS:
        out = pattern.sub(f"[{name}]", out)
    return out


def redact_settings(settings: dict[str, str]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, value in settings.items():
        if key.strip().lower() in SENSITIVE_SETTING_KEYS:
            clean[key] = "[REDACTED]"
        else:
            clean[key] = redact(value)
    return clean
