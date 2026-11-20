# Design notes

Notes on the trade-offs behind FixPilot. Useful when explaining the system end to end.

## 1. Request lifecycle

1. **Intake.** The client (web app, Teams bot or ADK chat) uploads a photo and an optional
   message. The API checks MIME type and size (8 MB max).
2. **Perception (`analyze_screen`).** Gemini receives the image plus a prompt that
   constrains `screen_type` to an enum and asks for a calibrated confidence. Structured
   output (`response_schema=ScreenAnalysis`) means the response is validated by Pydantic
   or rejected. Text and settings are redacted, and the raw image is removed from state.
3. **Confidence gate.** Below `FIXPILOT_MIN_VISION_CONFIDENCE` (default 0.6), or
   `unknown`, the agent asks for a better photo, up to 2 times, then escalates.
4. **Intent (`classify_intent` / `ask_intent`).** The model may only use the user's words.
   If the goal is unclear, the graph pauses with `interrupt()` and offers options. After 2
   unclear answers it escalates.
5. **Retrieval.** The query combines screen type, model, visible text, intent and the user's
   words; results are filtered by vendor. Locally this is BM25; in production it is
   Vertex AI RAG Engine (text-embedding-005, 512-token chunks, 64 overlap).
6. **Planning.** Candidate playbooks are filtered by intent, screen type and vendor. The
   LLM picks one (or none) and writes a short explanation. A hallucinated id is treated as
   "none", which escalates.
7. **Risk gate.** Pure code. For each step, the registry supplies risk, reversibility,
   allowed intents and prerequisites:
   * an action not allowed for the intent (e.g. `disable_secure_boot` when the user just
     wants Windows to start) leads to escalation;
   * CRITICAL actions (reset, clear TPM) always escalate;
   * missing prerequisites (BitLocker key) lead to a question;
   * HIGH actions need explicit confirmation, with the undo path shown.
8. **Guided execution.** One step per turn. The user replies "done", sends a photo, or asks
   for help. Photos are compared with `expected_screen`; mismatches re-show the step with an
   explanation, and more than 2 mismatches escalate.
9. **Escalation.** A ServiceNow incident is created with the redacted screen analysis,
   intent, conversation and current step, so the technician starts with full context.

## 2. Why LangGraph *and* ADK?

* **LangGraph** is used for the part that needs deterministic control flow: explicit
  states, conditional edges, `interrupt()`/`Command(resume=...)` for human-in-the-loop, and
  checkpointing so a session survives across HTTP requests.
* **ADK** is used for the conversational shell and Google-native deployment: `adk web` for
  local testing, `adk deploy agent_engine` for a managed runtime, and composition with other
  ADK agents. The ADK agent's instruction forbids it from inventing steps; it only relays
  what the workflow returns.

## 3. Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Vision misreads the screen | Enum-constrained output, confidence threshold, re-photo request, per-step photo verification |
| Model assumes intent | Intent prompt forbids screen-only inference; clarification node; eval metric `clarification_recall` = 1.0 |
| Model invents steps | Steps only from vetted playbooks; unknown playbook id escalates |
| Dangerous step suggested | Code-based risk registry; intent allow-lists; prerequisites; explicit confirmation; CRITICAL is human-only |
| Duplicate side effects on resume | One interrupt per node; no side effects before an interrupt; ticket creation in a non-interrupting node |
| PII leakage | Redaction before state/logs/tickets; images not persisted; fixtures use fake identifiers |
| Regressions | Offline eval gate in Cloud Build blocks deploys |

## 4. Scaling considerations

* Cloud Run autoscaling; a stateless API once the checkpointer is shared (Postgres).
* Vision is the dominant cost and latency. Options include a smaller Flash model for
  `analyze_screen`, image downscaling before upload, and caching the classification of
  repeated photos by perceptual hash.
* Playbook coverage scales with content, not code: new vendors need YAML + KB docs + eval
  cases, and the registry test enforces that every action is risk-rated.

## 5. What I would measure in production

* Resolution rate without escalation, by playbook.
* Escalation reasons (unclear intent vs. low vision confidence vs. policy block).
* Step-mismatch rate per step (shows confusing instructions).
* Time-to-resolution and token cost per resolved session.
* Zero tolerance: any HIGH-risk step shown without confirmation (logged in the trace).
