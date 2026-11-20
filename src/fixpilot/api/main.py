"""FastAPI service (deployed to Cloud Run).

POST /v1/sessions                 multipart: message, image  -> first agent turn
POST /v1/sessions/{id}/reply      multipart: text, choice, image -> next agent turn
GET  /v1/sessions/{id}/trace      node-by-node trace for debugging / audit
GET  /healthz
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .. import __version__
from ..graph import FixPilot
from ..observability import configure_logging

configure_logging()
log = logging.getLogger("fixpilot.api")

MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/heic", "application/json"}

app = FastAPI(title="FixPilot", version=__version__)
agent = FixPilot()


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    started = time.perf_counter()
    response = await call_next(request)
    log.info("request", extra={"request_id": rid, "path": request.url.path,
                               "status": response.status_code,
                               "latency_ms": round((time.perf_counter() - started) * 1000)})
    response.headers["x-request-id"] = rid
    return response


async def _read_image(image: UploadFile | None) -> tuple[bytes | None, str]:
    if image is None:
        return None, "image/jpeg"
    mime = image.content_type or "image/jpeg"
    if mime not in ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported image type {mime}")
    data = await image.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (max 8 MB)")
    return data, mime


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": __version__}


@app.post("/v1/sessions")
async def create_session(message: str = Form(""), image: UploadFile | None = File(None)):
    data, mime = await _read_image(image)
    if not data and not message:
        raise HTTPException(400, "Send a photo of the screen and/or a message")
    return agent.start(message=message, image=data, mime_type=mime)


@app.post("/v1/sessions/{session_id}/reply")
async def reply(session_id: str, text: str | None = Form(None), choice: str | None = Form(None),
                image: UploadFile | None = File(None)):
    data, mime = await _read_image(image)
    try:
        return agent.reply(session_id, text=text, choice=choice, image=data, mime_type=mime)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "Unknown or finished session") from exc


@app.get("/v1/sessions/{session_id}/trace")
def trace(session_id: str):
    return {"session_id": session_id, "trace": agent.trace(session_id)}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled_error")
    return JSONResponse({"error": "internal_error"}, status_code=500)
