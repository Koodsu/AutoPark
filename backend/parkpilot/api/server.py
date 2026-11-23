"""FastAPI application (server glue; excluded from the coverage target)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from parkpilot.api.frames import scenario_summary
from parkpilot.api.protocol import (
    PROTOCOL_VERSION,
    ConfigResponse,
    FavoriteView,
    LocationPreferenceView,
    RunInfo,
    ScenarioSummary,
    StartRunRequest,
)
from parkpilot.api.session import SessionManager
from parkpilot.core.config import ParkPilotConfig, UserSettings
from parkpilot.preferences.store import JsonFilePreferenceStore, PreferenceRepository
from parkpilot.scenarios.loader import list_scenario_ids, list_scenarios
from parkpilot.sim.systems import SYSTEMS


def create_app(preferences: PreferenceRepository | None = None) -> FastAPI:
    prefs = preferences or JsonFilePreferenceStore(
        Path(os.environ.get("PARKPILOT_PREFS", ".parkpilot/preferences.json"))
    )
    manager = SessionManager(prefs)
    app = FastAPI(title="ParkPilot", version=str(PROTOCOL_VERSION))
    app.state.manager = manager

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/scenarios")
    def scenarios() -> list[ScenarioSummary]:
        return [scenario_summary(s) for s in list_scenarios()]

    @app.get("/api/config")
    def config() -> ConfigResponse:
        return ConfigResponse(
            protocol_version=PROTOCOL_VERSION,
            systems=sorted(SYSTEMS),
            config=ParkPilotConfig().model_dump(mode="json"),
        )

    @app.post("/api/runs")
    def start_run(req: StartRunRequest) -> RunInfo:
        if req.scenario_id not in list_scenario_ids():
            raise HTTPException(404, f"unknown scenario {req.scenario_id}")
        session = manager.start(req.scenario_id, req.system, req.seed, req.settings)
        return RunInfo(
            run_id=session.run_id,
            ws_path=f"/ws/runs/{session.run_id}",
            scenario_id=req.scenario_id,
            system=req.system,
        )

    @app.delete("/api/runs/{run_id}")
    def stop_run(run_id: str) -> dict[str, bool]:
        if not manager.stop(run_id):
            raise HTTPException(404, "unknown run")
        return {"stopped": True}

    @app.get("/api/preferences")
    def get_preferences() -> UserSettings:
        return prefs.get_settings()

    @app.put("/api/preferences")
    def put_preferences(settings: UserSettings) -> UserSettings:
        prefs.save_settings(settings)
        return settings

    @app.get("/api/preferences/locations")
    def locations() -> list[LocationPreferenceView]:
        return [
            LocationPreferenceView(
                location_id=loc.location_id,
                favorites=[FavoriteView(x=f.x, y=f.y, count=f.count) for f in loc.favorites],
            )
            for loc in prefs.list_locations()
        ]

    @app.delete("/api/preferences/locations/{location_id}")
    def clear_location(location_id: str) -> dict[str, bool]:
        prefs.clear_location(location_id)
        return {"cleared": True}

    @app.websocket("/ws/runs/{run_id}")
    async def run_socket(ws: WebSocket, run_id: str) -> None:
        session = manager.sessions.get(run_id)
        await ws.accept()
        if session is None:
            await ws.close(code=4404)
            return
        try:
            await manager.stream(ws, session)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
