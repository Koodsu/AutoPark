# ParkPilot

A 2D simulated autonomous parking stack, built to show measurable engineering responses
to the most common owner complaints about consumer "self-driving" parking features:
indecisive shuffling, crawling, crooked parking, bad spot choice, parking in restricted
places, close-range blindness, phantom braking and more. It is a research simulator, not a
product, and has no affiliation with any manufacturer.

- **Backend** (Python 3.12, numpy, pydantic, FastAPI): simulated camera-only perception
  with an occupancy grid that remembers, lined and line-less spot detection with legality
  rules, destination-aware explainable ranking, Hybrid A* with Reeds-Shepp, final-pose
  refinement, a safety supervisor, and a decision state machine with commitment and a
  fallback ladder.
- **Frontend** (TypeScript, React, Canvas 2D): live HMI with the chosen spot, ranked
  alternatives and reasons, the planned path, the current state and why, clearance
  gauges, click-to-override, pause/abort/speed and a baseline toggle.

Motivation and the complaint-to-capability map: [docs/PROBLEM_STATEMENT.md](docs/PROBLEM_STATEMENT.md).

![Garage with pillars](docs/img/garage_pillars.png)

## Before / after

`make bench` runs all 16 scenarios x 5 seeds for ParkPilot and a deliberately naive
baseline (first visible spot, no legality checks, no gear-change penalty, timed
re-planning, memoryless single-frame perception, no persistence filter, constant crawl,
line-centred nose-in, no fallback). Full report: [docs/BENCHMARK.md](docs/BENCHMARK.md).

| Metric | Baseline | ParkPilot |
|---|---:|---:|
| Park success rate (excl. expected handbacks) | 64% | 100% |
| Collisions (total) | 7 | 0 |
| Curb contacts (total) | 18 | 0 |
| Legality violations (total) | 10 | 0 |
| Missed hazards (total) | 0 | 0 |
| False stops (total) | 764 | 35 |
| Gear changes per park | 8.5 | 2.3 |
| Re-plans per park | 19.0 | 0.1 |
| Time to park (s) | 59.9 | 46.8 |
| Mean speed (m/s) | 0.60 | 1.19 |
| Peak jerk, raw (m/s^3) | 13.9 | 8.3 |
| Peak jerk excl. emergency stops (m/s^3) | 4.6 | 2.4 |
| Final lateral error (m) | 0.10 | 0.04 |
| Final heading error (deg) | 3.0 | 0.6 |
| Min side clearance at rest (m) | 0.93 | 0.84 |
| Walking distance (m) | 16.6 | 12.7 |
| Selection matches oracle | 30% | 100% |
| Abort/handback with reason | 0% | 100% |

Across 160 runs ParkPilot parked in every run where parking was the right outcome, with 0
collisions, 0 curb contacts, 0 legality violations and 0 missed hazards; every final
heading is within 1.5 deg and every charge port within 0.5 m of its charger. "Min side clearance at rest" is lower for ParkPilot only because the
baseline mostly ends up in roomy stalls far from the entrance.

![The baseline in dance_spot_opens: memoryless perception, it stops for every flicker](docs/img/baseline_memoryless.png)

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/), Node 22 with pnpm, and Chromium for the
end-to-end test.

```bash
make install     # uv sync + pnpm install
make dev         # backend on :8000, Vite on :5173 (proxying /api and /ws)
# open http://localhost:5173, pick a scenario (e.g. garage_pillars, charger_stall,
# unmarked_lot) and press Start
make test        # backend pytest (coverage >= 85%) + frontend vitest
make lint        # ruff, ruff format, mypy --strict, tsc, eslint, prettier
make bench       # full A/B benchmark -> docs/BENCHMARK.md and docs/img/bench_*.png
make e2e         # Playwright: start both servers, park in charger_stall
make schema      # export the protocol JSON Schema and regenerate TS types
```

CLI: `uv run parkpilot list | serve | bench | export-schema` (from `backend/`).

## Controls

| Control | What it does |
|---|---|
| Scenario, ParkPilot/Baseline, seed, Start | start a run of either system on any scenario |
| Click a spot (map or list) | override the choice; the car acknowledges and re-plans, and the choice is remembered for that location |
| Pause / Resume, speed | pause the sim or run it at 0.5x-8x |
| Abort | stop safely and hand back control |
| Ground truth | overlay true obstacles, agents and zones (debug) |
| Preferences | entry direction (auto, nose-in, reverse-in, pull-through), unlocks (placard, charging, expectant, permit), ranking weights |
| Mouse drag / wheel | pan / zoom |

Map legend: green = eligible (with rank), blue = chosen, red X = restricted, amber ? =
not yet verified, grey = occupied, dashed = line-less (inferred) stall; blue/orange path
= forward/reverse; red circles = confirmed moving agents.

![Charger stall](docs/img/charger_stall.png)

## Architecture

Python backend under `backend/parkpilot/` with the dependency direction
`core <- vehicle/world <- perception <- spots/planning <- behavior <- sim <- api`; the
planner and behavior layers only see perception output, never ground truth. The browser
frontend renders frames and sends commands; it has no autonomy logic. The pydantic wire
protocol is exported as JSON Schema and compiled to TypeScript, and CI fails on drift.
Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/PROTOCOL.md](docs/PROTOCOL.md),
and the ADRs in [docs/adr/](docs/adr/).

![Yielding to a pedestrian](docs/img/dynamic_agents_yield.png)

## Known limitations

- 2D only: no heights beyond a "low object" flag, no ramps, no 3D garage structure.
- Simulated perception: ray-cast depth with noise models, not real images; painted-line
  observations carry their identity (data association is simulated). No real sensor data.
- Kinematic bicycle model, not a dynamic one (no tyre slip, no load transfer).
- Ego localization is assumed perfect.
- Planning latency is not simulated (the sim pauses while Hybrid A* runs).
- Line-less inference only offers gaps between two parked vehicles, not open row ends.
- Pure Python: Hybrid A* runs ~1-2 ms per expansion. In tight aisles it can exhaust its
  budget; it then retries with 3x the budget and a finer steering set before the fallback
  ladder moves on (garage_pillars now parks 5/5).
- The baseline re-plans on a 2 s timer rather than every perception frame (every frame
  would be slower to benchmark and only worse for the baseline).

## Where ParkPilot still underperforms

- `dynamic_agents`: about 18 s lost yielding per run. The car waits until predicted
  conflicts clear, including for a car that backs out and then drives slowly ahead of it.
  A car-following mode was tried; it cut yield time to ~15 s but made time-to-park worse
  (more gear changes behind a slow car), so it was not kept.
- Raw peak jerk reaches 40 m/s^3 (the emergency brake build-up limit) in scenarios where
  a pedestrian steps out from between parked cars and is confirmed late. Peak jerk
  excluding those genuine emergency stops is ~2-5 m/s^3.
- 35 false stops remain over 80 runs (the baseline has 764). They are brief yields to
  clutter that looks like a moving object for a few frames.
- Charger alignment depends on mounting: with the port 0.33 m inside the rear bumper and
  no bumper overhang in the 2D model, a pedestal *behind* the stall head cannot get within
  0.5 m. The scenarios use pedestals on the stall line and a wall box on the garage side
  wall, as many installations do.
- REFINE corrections add time and gear changes in tight or noisy (night) spots
  (night_bollards: up to 7 gear changes in one seed).
