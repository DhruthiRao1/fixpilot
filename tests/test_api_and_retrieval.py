from fastapi.testclient import TestClient

from fixpilot.api.main import app
from fixpilot.config import get_settings
from fixpilot.retrieval.local import LocalKeywordRetriever


def test_retriever_vendor_filter_and_ranking():
    r = LocalKeywordRetriever(get_settings().kb_dir)
    hits = r.search("operating system boot mode change pass code", vendor="HP")
    assert hits and hits[0].doc_id == "hp_secure_boot_code"
    dell = r.search("operating system boot mode change pass code", vendor="Dell")
    assert all(h.metadata.get("vendor", "any").lower() != "hp" for h in dell)


def test_api_session_roundtrip(screen):
    client = TestClient(app)
    assert client.get("/healthz").json()["ok"]
    res = client.post("/v1/sessions", data={"message": ""},
                      files={"image": ("s.json", screen("hp_boot_menu"), "application/json")})
    body = res.json()
    assert res.status_code == 200 and body["prompt"]["type"] == "question"
    res = client.post(f"/v1/sessions/{body['session_id']}/reply",
                      data={"choice": "start_normally"})
    assert res.json()["playbook"] == "boot_menu_continue_normally"
    trace = client.get(f"/v1/sessions/{body['session_id']}/trace").json()["trace"]
    assert [t["node"] for t in trace][:2] == ["analyze_screen", "classify_intent"]


def test_api_rejects_bad_mime_and_unknown_session():
    client = TestClient(app)
    r = client.post("/v1/sessions", files={"image": ("x.gif", b"GIF89a", "image/gif")})
    assert r.status_code == 415
    assert client.post("/v1/sessions/nope/reply", data={"text": "hi"}).status_code == 404
