import base64
import contextlib
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from parkpilot.api.frames import build_frame, build_scene
from parkpilot.api.protocol import (
    PROTOCOL_VERSION,
    ClientCommand,
    FrameMessage,
    ServerMessage,
    export_schema,
)
from parkpilot.api.server import create_app
from parkpilot.preferences.store import InMemoryPreferenceStore
from parkpilot.scenarios.loader import load_scenario
from parkpilot.sim.engine import Simulation
from parkpilot.sim.systems import SYSTEMS, system_config

SERVER = TypeAdapter(ServerMessage)


def _sim() -> Simulation:
    return Simulation(
        load_scenario("restricted_spots"), SYSTEMS["parkpilot"], system_config("parkpilot"), 0
    )


def test_frame_round_trips_through_the_protocol() -> None:
    sim = _sim()
    sim.run(max_ticks=200)
    frame = build_frame(sim, seq=1, system="parkpilot", include_grid=True, debug=True)
    raw = frame.model_dump_json()
    parsed = SERVER.validate_json(raw)
    assert isinstance(parsed, FrameMessage)
    assert parsed.protocol_version == PROTOCOL_VERSION
    assert parsed.spots and parsed.spots[0].rank == 1
    assert parsed.debug is not None and parsed.debug.obstacles
    assert parsed.grid is not None
    g = sim.perception.grid
    data = np.frombuffer(base64.b64decode(parsed.grid.data), dtype=np.uint8)
    assert data.size == g.nx * g.ny
    assert (data > 0).sum() == g.observed.sum()
    scene = build_scene(sim, "r1", "parkpilot")
    assert scene.grid.nx == g.nx and scene.scenario.id == "restricted_spots"
    lean = build_frame(sim, seq=2, system="parkpilot")
    assert lean.grid is None and lean.debug is None


def test_schema_export_covers_all_messages() -> None:
    schema = json.loads(export_schema())
    defs = schema["$defs"]
    for name in ("FrameMessage", "SceneMessage", "OverrideSpotCommand", "SetSpeedCommand"):
        assert name in defs
    cmd = TypeAdapter(ClientCommand).validate_python({"type": "set_speed", "factor": 2})
    assert cmd.type == "set_speed"


def test_rest_and_websocket(tmp_path: Path) -> None:
    prefs = InMemoryPreferenceStore()
    client = TestClient(create_app(prefs))
    assert client.get("/api/health").json() == {"status": "ok"}
    ids = [s["id"] for s in client.get("/api/scenarios").json()]
    assert "garage_pillars" in ids and "charger_stall" in ids
    cfg = client.get("/api/config").json()
    assert cfg["protocol_version"] == PROTOCOL_VERSION and "baseline" in cfg["systems"]
    assert client.post("/api/runs", json={"scenario_id": "nope"}).status_code == 404
    run = client.post("/api/runs", json={"scenario_id": "restricted_spots", "seed": 1}).json()
    with client.websocket_connect(run["ws_path"]) as ws:
        scene = json.loads(ws.receive_text())
        assert scene["type"] == "scene" and scene["run_id"] == run["run_id"]
        frame = json.loads(ws.receive_text())
        assert frame["type"] == "frame"
        ws.send_text(json.dumps({"type": "set_speed", "factor": 8}))
        ws.send_text(json.dumps({"type": "set_debug", "enabled": True}))
        ws.send_text(json.dumps({"type": "pause"}))
        ws.send_text(json.dumps({"type": "bogus"}))
        acks: list[str] = []
        for _ in range(200):
            msg = json.loads(ws.receive_text())
            if msg["type"] in ("ack", "error"):
                acks.append(msg.get("message", ""))
            if len(acks) == 4:
                break
        assert acks[:3] == ["speed x8", "debug on", "paused"]
        assert acks[3].startswith("bad command")
        ws.send_text(json.dumps({"type": "override_spot", "spot_id": "missing"}))
        ws.send_text(json.dumps({"type": "resume"}))
        ws.send_text(json.dumps({"type": "abort"}))
        seen = []
        for _ in range(300):
            msg = json.loads(ws.receive_text())
            if msg["type"] == "ack":
                seen.append(msg["message"])
            if len(seen) == 3:
                break
        assert "not known" in seen[0] and seen[1] == "resumed" and seen[2].startswith("aborting")
    assert client.delete(f"/api/runs/{run['run_id']}").json() == {"stopped": True}
    assert client.delete("/api/runs/unknown").status_code == 404
    prefs.record_override("office_park", 1.0, 2.0)
    locs = client.get("/api/preferences/locations").json()
    assert locs[0]["location_id"] == "office_park"
    assert client.delete("/api/preferences/locations/office_park").json() == {"cleared": True}
    put = client.put("/api/preferences", json={"direction": "nose_in", "unlocks": ["placard"]})
    assert put.status_code == 200
    assert client.get("/api/preferences").json()["direction"] == "nose_in"
    with client.websocket_connect("/ws/runs/unknown") as ws, contextlib.suppress(Exception):
        ws.receive_text()  # closed with code 4404
