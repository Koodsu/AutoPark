"""Pure functions that turn simulation state into wire messages."""

from __future__ import annotations

import base64

import numpy as np

from parkpilot.api.protocol import (
    DebugLayer,
    DebugObject,
    EgoView,
    FrameMessage,
    GridInfo,
    GridPatch,
    LiveMetrics,
    PoseModel,
    ScenarioSummary,
    SceneMessage,
    SpotView,
    TrackView,
    TransitionView,
    VehicleInfo,
)
from parkpilot.core.geometry import FloatArray, Pose
from parkpilot.metrics.run_metrics import gear_changes
from parkpilot.safety.supervisor import track_kind
from parkpilot.scenarios.schema import Scenario
from parkpilot.sim.engine import Simulation

MAX_PATH_POINTS = 400
MAX_SPOTS = 60


def _poly(p: FloatArray) -> list[tuple[float, float]]:
    return [(round(float(x), 3), round(float(y), 3)) for x, y in p]


def _pose(p: Pose) -> PoseModel:
    return PoseModel(x=p.x, y=p.y, theta=p.theta)


def scenario_summary(sc: Scenario) -> ScenarioSummary:
    return ScenarioSummary(
        id=sc.id,
        title=sc.title,
        complaints=list(sc.complaints),
        description=sc.description.strip(),
        location_id=sc.location_id,
        lighting=sc.lighting,
    )


def build_scene(sim: Simulation, run_id: str, system: str) -> SceneMessage:
    g = sim.perception.grid
    v = sim.config.vehicle
    return SceneMessage(
        run_id=run_id,
        system=system,
        scenario=scenario_summary(sim.scenario),
        bounds=sim.scenario.bounds,
        entrance=sim.scenario.entrance,
        grid=GridInfo(
            origin_x=float(g.origin[0]),
            origin_y=float(g.origin[1]),
            resolution=g.res,
            nx=g.nx,
            ny=g.ny,
        ),
        vehicle=VehicleInfo(
            length=v.length,
            width=v.width,
            mirror_width=v.mirror_width,
            wheelbase=v.wheelbase,
            rear_overhang=v.rear_overhang,
        ),
    )


def encode_grid(sim: Simulation) -> GridPatch:
    """0 = never observed; otherwise 1..255 maps occupancy probability."""
    g = sim.perception.grid
    p = g.probability()
    data = np.where(g.observed, 1 + np.round(p * 254), 0).astype(np.uint8)
    return GridPatch(version=g.version, data=base64.b64encode(data.tobytes()).decode("ascii"))


def build_frame(
    sim: Simulation,
    *,
    seq: int,
    system: str,
    paused: bool = False,
    speed_factor: float = 1.0,
    include_grid: bool = False,
    debug: bool = False,
) -> FrameMessage:
    tel = sim.autonomy.telemetry
    st = sim.state
    path: list[tuple[float, float, int]] = []
    if tel.path is not None and len(tel.path.poses):
        poses, dirs = tel.path.poses, tel.path.directions
        step = max(1, len(poses) // MAX_PATH_POINTS)
        path = [
            (round(float(p[0]), 3), round(float(p[1]), 3), int(d))
            for p, d in zip(poses[::step], dirs[::step], strict=True)
        ]
    spots = []
    for rank, r in enumerate(tel.ranked[:MAX_SPOTS]):
        c = r.candidate
        spots.append(
            SpotView(
                id=c.id,
                polygon=_poly(c.polygon()),
                layout=c.layout.value,
                spot_type=c.spot_type.value,
                legal=c.legal,
                occupied=c.occupied,
                eligible=r.eligible,
                rank=rank + 1,
                cost=round(r.cost, 2),
                reasons=r.reasons[:4],
                chosen=c.id == tel.chosen_spot,
                source=c.source,
            )
        )
    tracks = []
    perc = sim.last_perception
    if perc is not None:
        trk_cfg = sim.config.tracker
        for tr in perc.tracks:
            if tr.extent > 3.0:
                continue
            tracks.append(
                TrackView(
                    id=tr.id,
                    x=float(tr.position[0]),
                    y=float(tr.position[1]),
                    vx=float(tr.velocity[0]),
                    vy=float(tr.velocity[1]),
                    extent=tr.extent,
                    confirmed=tr.confirmed(trk_cfg),
                    dynamic=tr.is_dynamic(trk_cfg),
                    kind=track_kind(tr),
                )
            )
    shifts = gear_changes(np.array([r.speed for r in sim.log.ticks]))
    return FrameMessage(
        seq=seq,
        t=round(sim.t, 3),
        paused=paused,
        speed_factor=speed_factor,
        system=system,
        state=tel.state.value,
        reason=tel.reason,
        ego=EgoView(
            pose=_pose(st.pose),
            speed=st.speed,
            steer=st.steer,
            footprint=[_poly(p) for p in sim.vehicle.footprint(st.pose)],
        ),
        chosen_spot=tel.chosen_spot,
        goal=_pose(tel.goal.pose) if tel.goal is not None else None,
        path=path,
        spots=spots,
        tracks=tracks,
        grid=encode_grid(sim) if include_grid else None,
        clearances={k: round(v, 2) for k, v in tel.clearances.items()},
        metrics=LiveMetrics(
            elapsed=round(sim.t, 2),
            speed=round(abs(st.speed), 2),
            gear_changes=shifts,
            replans=tel.replans,
            spot_switches=tel.spot_switches,
            yield_events=tel.yield_events,
            hard_brakes=tel.hard_brakes,
        ),
        transitions=[
            TransitionView(t=round(x.t, 2), src=x.src.value, dst=x.dst.value, reason=x.reason)
            for x in tel.transitions[-12:]
        ],
        safety_reason=tel.safety.reason if tel.safety is not None else "",
        handback_reason=tel.handback_reason,
        override_ack=tel.override_ack,
        done=sim.done,
        debug=build_debug(sim) if debug else None,
    )


def build_debug(sim: Simulation) -> DebugLayer:
    w = sim.world
    return DebugLayer(
        obstacles=[
            DebugObject(id=o.id, kind=o.kind.value, polygon=_poly(o.polygon)) for o in w.obstacles
        ],
        agents=[
            DebugObject(id=a.id, kind=a.spec.kind, polygon=_poly(a.polygon()))
            for a in sim.agents.active()
        ],
        spots=[
            DebugObject(id=s.id, kind=s.spot_type.value, polygon=_poly(s.polygon()))
            for s in w.spots
        ],
        zones=[
            DebugObject(id=z.id, kind=z.zone_type.value, polygon=_poly(z.polygon)) for z in w.zones
        ],
    )
