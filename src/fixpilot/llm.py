"""Model layer.

Two interchangeable backends behind one interface:

* ``GeminiLLM`` - Gemini on Vertex AI via the ``google-genai`` SDK, using structured
  (JSON-schema) output so every response is validated by Pydantic.
* ``MockLLM``  - deterministic and offline. Used in unit tests, CI and the eval gate so
  the graph logic can be tested without network calls or cost. Its "images" are JSON
  fixtures describing a screen (see evals/fixtures).
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from pydantic import BaseModel

from .config import Settings, get_settings
from .schemas import Intent, IntentResult, RetrievedDoc, ScreenAnalysis, ScreenType

log = logging.getLogger(__name__)


class PlaybookChoice(BaseModel):
    playbook_id: str | None
    explanation: str


class LLM(Protocol):
    def analyze_screen(self, image: bytes, mime_type: str) -> ScreenAnalysis: ...

    def classify_intent(self, message: str, screen: ScreenAnalysis) -> IntentResult: ...

    def choose_playbook(
        self,
        intent: Intent,
        screen: ScreenAnalysis,
        candidates: list[dict],
        docs: list[RetrievedDoc],
    ) -> PlaybookChoice: ...


# --------------------------------------------------------------------------- prompts

VISION_PROMPT = """You are an IT support vision model. Describe the screen in the photo.
Classify screen_type using ONLY these values: {types}.
Extract vendor and model if visible, the key on-screen text, any visible settings as
name -> value, and the selectable options. Do NOT copy serial numbers, UUIDs, emails or
product keys. Set confidence (0-1) to how sure you are of screen_type; if the photo is
blurry, cropped or ambiguous, use a low confidence rather than guessing."""

INTENT_PROMPT = """Decide what the user is trying to achieve. Allowed intents: {intents}.
Return "unknown" unless the user's own words make the goal clear. Never infer intent
from the screen alone: the same boot menu can mean "boot from USB" or "I pressed a key
by accident". Screen context: {screen}
User message: {message!r}"""

PLAYBOOK_PROMPT = """Pick the single best playbook for this user, or null if none fits.
Intent: {intent}. Screen: {screen}
Candidate playbooks: {candidates}
Reference documentation:
{docs}
Explain in one or two plain sentences why this fixes the user's problem."""


# --------------------------------------------------------------------------- Gemini


class GeminiLLM:
    def __init__(self, settings: Settings | None = None):
        from google import genai  # imported lazily so mock mode needs no credentials

        self.settings = settings or get_settings()
        if not self.settings.gcp_project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set for the gemini backend")
        self._client = genai.Client(
            vertexai=True,
            project=self.settings.gcp_project,
            location=self.settings.gcp_location,
        )

    def _generate(self, contents, schema: type[BaseModel]) -> BaseModel:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.settings.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return schema.model_validate_json(resp.text)

    def analyze_screen(self, image: bytes, mime_type: str) -> ScreenAnalysis:
        from google.genai import types

        prompt = VISION_PROMPT.format(types=[t.value for t in ScreenType])
        part = types.Part.from_bytes(data=image, mime_type=mime_type)
        return self._generate([part, prompt], ScreenAnalysis)  # type: ignore[return-value]

    def classify_intent(self, message: str, screen: ScreenAnalysis) -> IntentResult:
        prompt = INTENT_PROMPT.format(
            intents=[i.value for i in Intent],
            screen=screen.model_dump_json(include={"screen_type", "vendor", "model"}),
            message=message,
        )
        return self._generate(prompt, IntentResult)  # type: ignore[return-value]

    def choose_playbook(self, intent, screen, candidates, docs) -> PlaybookChoice:
        prompt = PLAYBOOK_PROMPT.format(
            intent=intent.value,
            screen=screen.model_dump_json(include={"screen_type", "vendor", "model", "settings"}),
            candidates=json.dumps(candidates),
            docs="\n---\n".join(f"[{d.doc_id}] {d.text[:1500]}" for d in docs) or "(none)",
        )
        choice: PlaybookChoice = self._generate(prompt, PlaybookChoice)  # type: ignore[assignment]
        valid = {c["id"] for c in candidates}
        if choice.playbook_id not in valid:  # hallucinated id -> treat as no match
            log.warning("model chose unknown playbook %r", choice.playbook_id)
            choice.playbook_id = None
        return choice


# --------------------------------------------------------------------------- Mock

_INTENT_KEYWORDS: list[tuple[Intent, tuple[str, ...]]] = [
    (Intent.RESET_DEVICE, ("reset", "reinstall", "wipe", "factory", "format")),
    (Intent.RESTORE_SETTINGS, ("undo", "turn back on", "put back", "restore", "revert",
                               "enable secure boot", "back to normal settings")),
    (Intent.BOOT_FROM_USB, ("usb", "pendrive", "pen drive", "flash drive", "install linux",
                            "install windows", "bootable", "ubuntu", "rufus")),
    (Intent.START_NORMALLY, ("just turned", "switched on", "switch on", "switched it on",
                             "turned it on", "nothing i did",
                             "didn't do anything", "did nothing", "start normally",
                             "start windows", "cleaning", "cleaned", "accident", "by mistake",
                             "just want", "normal")),
    (Intent.FIX_ERROR, ("error", "blue screen", "crash", "stop code", "not working")),
]


class MockLLM:
    """Deterministic stand-in for Gemini (tests, CI, offline demos)."""

    def analyze_screen(self, image: bytes, mime_type: str) -> ScreenAnalysis:
        try:
            data = json.loads(image.decode("utf-8"))
            return ScreenAnalysis.model_validate(data)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            # A real photo in mock mode: we genuinely cannot read it.
            return ScreenAnalysis(screen_type=ScreenType.UNKNOWN, confidence=0.0)

    def classify_intent(self, message: str, screen: ScreenAnalysis) -> IntentResult:
        text = (message or "").lower()
        for intent, words in _INTENT_KEYWORDS:
            hit = next((w for w in words if w in text), None)
            if hit:
                return IntentResult(intent=intent, confidence=0.9, rationale=f"matched '{hit}'")
        return IntentResult(intent=Intent.UNKNOWN, confidence=0.0, rationale="no clear goal")

    def choose_playbook(self, intent, screen, candidates, docs) -> PlaybookChoice:
        if not candidates:
            return PlaybookChoice(playbook_id=None, explanation="No vetted playbook matches.")
        best = candidates[0]
        return PlaybookChoice(playbook_id=best["id"], explanation=best["summary"])


def build_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    if settings.llm_backend == "gemini":
        return GeminiLLM(settings)
    if settings.llm_backend == "mock":
        return MockLLM()
    raise ValueError(f"Unknown FIXPILOT_LLM_BACKEND={settings.llm_backend!r}")
