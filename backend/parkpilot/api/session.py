"""WebSocket run sessions: a simulation stepped in real time (times a speed factor)."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field

from fastapi import WebSocket
from pydantic import TypeAdapter

from parkpilot.api.frames import build_frame, build_scene
from parkpilot.api.protocol import (
    AbortCommand,
    AckMessage,
    ClientCommand,
    ErrorMessage,
    OverrideSpotCommand,
    PauseCommand,
    ResumeCommand,
    SetDebugCommand,
    SetSpeedCommand,
)
from parkpilot.behavior.types import OverrideSpot
from parkpilot.core.config import UserSettings
from parkpilot.preferences.store import PreferenceRepository
from parkpilot.scenarios.loader import load_scenario
from parkpilot.sim.engine import DT, Simulation
from parkpilot.sim.systems import SYSTEMS, system_config

FRAME_HZ = 30.0
GRID_HZ = 4.0
COMMANDS: TypeAdapter[ClientCommand] = TypeAdapter(ClientCommand)


@dataclass
class RunSession:
    run_id: str
    system: str
    sim: Simulation
    paused: bool = False
    speed_factor: float = 1.0
    debug: bool = False
    seq: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def advance(self, wall_dt: float) -> None:
        if self.paused or self.sim.done:
            return
        n = max(1, round(wall_dt * self.speed_factor / DT))
        async with self._lock:
            await asyncio.to_thread(self._steps, n)

    def _steps(self, n: int) -> None:
        for _ in range(n):
            if self.sim.done:
                break
            self.sim.step()

    async def handle(self, raw: str) -> AckMessage | ErrorMessage:
        try:
            cmd = COMMANDS.validate_json(raw)
        except ValueError as exc:
            return ErrorMessage(message=f"bad command: {exc}")
        if isinstance(cmd, PauseCommand):
            self.paused = True
            return AckMessage(message="paused")
        if isinstance(cmd, ResumeCommand):
            self.paused = False
            return AckMessage(message="resumed")
        if isinstance(cmd, SetSpeedCommand):
            self.speed_factor = cmd.factor
            return AckMessage(message=f"speed x{cmd.factor:g}")
        if isinstance(cmd, SetDebugCommand):
            self.debug = cmd.enabled
            return AckMessage(message="debug on" if cmd.enabled else "debug off")
        async with self._lock:
            if isinstance(cmd, AbortCommand):
                return AckMessage(message=self.sim.abort())
            assert isinstance(cmd, OverrideSpotCommand)
            return AckMessage(message=self.sim.command(OverrideSpot(cmd.spot_id)))


class SessionManager:
    def __init__(self, preferences: PreferenceRepository) -> None:
        self.preferences = preferences
        self.sessions: dict[str, RunSession] = {}

    def start(
        self, scenario_id: str, system: str, seed: int, settings: UserSettings | None
    ) -> RunSession:
        sc = load_scenario(scenario_id)
        chosen = settings or self._merged_settings(sc.user)
        remembered = self.preferences.get_location(sc.location_id).positions()
        sim = Simulation(
            sc, SYSTEMS[system], system_config(system), seed, settings=chosen, remembered=remembered
        )
        location = sc.location_id

        def remember(x: float, y: float) -> None:
            self.preferences.record_override(location, x, y)

        sim.override_listeners.append(remember)
        run_id = uuid.uuid4().hex[:12]
        session = RunSession(run_id, system, sim)
        # Keep a bounded number of sessions alive.
        while len(self.sessions) >= 8:
            self.sessions.pop(next(iter(self.sessions)))
        self.sessions[run_id] = session
        return session

    def _merged_settings(self, scenario_settings: UserSettings) -> UserSettings:
        stored = self.preferences.get_settings()
        return stored.model_copy(
            update={"unlocks": stored.unlocks | scenario_settings.unlocks}
            | (
                {"direction": scenario_settings.direction}
                if scenario_settings.direction.value != "auto"
                else {}
            )
        )

    def stop(self, run_id: str) -> bool:
        return self.sessions.pop(run_id, None) is not None

    async def stream(self, ws: WebSocket, session: RunSession) -> None:
        await ws.send_text(
            build_scene(session.sim, session.run_id, session.system).model_dump_json()
        )
        receiver = asyncio.create_task(self._receive(ws, session))
        last = time.monotonic()
        last_grid = 0.0
        last_version = -1
        try:
            while not receiver.done():
                now = time.monotonic()
                await session.advance(now - last)
                last = now
                grid_due = now - last_grid >= 1.0 / GRID_HZ
                version = session.sim.perception.grid.version
                include_grid = grid_due and version != last_version
                if include_grid:
                    last_grid, last_version = now, version
                session.seq += 1
                frame = build_frame(
                    session.sim,
                    seq=session.seq,
                    system=session.system,
                    paused=session.paused,
                    speed_factor=session.speed_factor,
                    include_grid=include_grid,
                    debug=session.debug,
                )
                await ws.send_text(frame.model_dump_json())
                await asyncio.sleep(max(0.0, 1.0 / FRAME_HZ - (time.monotonic() - now)))
        finally:
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await receiver

    async def _receive(self, ws: WebSocket, session: RunSession) -> None:
        while True:
            raw = await ws.receive_text()
            reply = await session.handle(raw)
            await ws.send_text(reply.model_dump_json())
