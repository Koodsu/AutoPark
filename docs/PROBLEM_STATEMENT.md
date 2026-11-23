# Problem statement

ParkPilot is a research simulator. It exists to show, in a controlled and measurable 2D
setting, engineering responses to the most common owner complaints about consumer
"self-driving" parking features. The complaints below were collected (and fact-checked in
November 2026) from coverage of Tesla FSD destination parking, Autopark and Actually Smart
Summon, which are the most widely discussed systems of this kind. ParkPilot is not affiliated
with, and does not model the internals of, any commercial product.

## Background

- Tesla CEO Elon Musk has publicly said that destination parking is now by far the biggest
  reason people intervene with FSD ([Basenor][basenor], [InsideEVs][insideevs]).
- NHTSA opened preliminary evaluation **PE24033** in January 2025, covering about 2.6 million
  vehicles, into Actually Smart Summon after crashes in which cars failed to detect posts and
  parked vehicles ([Electrek][electrek-ass], [NHTSA ODI resume][nhtsa]). The probe was later
  closed without a recall after the agency reviewed about 159 incidents (97 crashes, no
  injuries).
- Tesla's 2022-23 removal of ultrasonic sensors in favor of vision-only Park Assist drew owner
  reports that close-range distance estimates were worse at night and on low objects such as
  curbs and snow piles ([Notebookcheck][notebookcheck]).
- Owners have asked for the system to respect accessible (handicap) spaces and other
  restricted spots ([Not a Tesla App][notateslaapp]), and for parking preferences such as
  reverse-in versus nose-in; a 2026 software update added parking preferences
  ([Electrek][electrek-prefs]).

**Caveat on sources.** Many of these complaints come from owner forums, social media and blog
aggregators that summarize them, not from controlled testing. The NHTSA document is the only
primary regulatory source. We treat the complaints as a catalogue of failure modes worth
designing against, not as statistics about any product.

## Complaints and engineering responses

Every complaint maps to a capability in the stack, at least one scenario that reproduces it,
and a metric that the benchmark (`make bench`, see [BENCHMARK.md](BENCHMARK.md)) reports for
both ParkPilot and a deliberately naive baseline.

### Decisiveness

| # | Complaint | Engineering response | Metric | Scenario(s) |
|---|---|---|---|---|
| C1 | Indecisive "parking dance": endless forward/reverse shuffling, re-planning mid-maneuver, hunting for a newly opened spot | The behavior state machine commits to a spot and only switches when a new spot is better by an absolute *and* relative margin, is reachable (a plan exists), and the maneuver has not begun. Hybrid A* adds a cost for each gear change, so plans minimize shuffles. The plan is only recomputed when it becomes invalid. | gear changes per park, re-plan count, time-to-park | `dance_spot_opens` |
| C2 | Painfully slow and creepy: crawls the whole way, then stops for no clear reason | The speed profile is driven by clearance from the distance field (fast in open space, slow near obstacles), limited by curvature and by jerk and acceleration limits, and decelerates smoothly to cusps and to the goal. | mean speed, time-to-park, peak jerk | `open_lot_speed` |
| C14 | Stops in the aisle or bails out with no fallback | A graceful degradation ladder: try the next-best spot, then circle the search route again, then stop safely at the curb and hand control back with a stated reason. | abort rate, abort-with-reason (must be 100%) | `full_lot_fallback` |

### Precision

| # | Complaint | Engineering response | Metric | Scenario(s) |
|---|---|---|---|---|
| C3 | Crooked or off-center final pose, too close to one neighbor | Final-pose refinement measures the *actual* neighbors in the occupancy grid and centers between them (within the painted lines), targeting door-opening clearance. A REFINE state corrects residual error with a short maneuver. | final lateral error, heading error, min side clearance | `crooked_neighbor` |
| C9 | Parallel parking failures | Reeds-Shepp analytic expansion inside Hybrid A* solves multi-cusp parallel maneuvers. The goal is offset to a target curb distance measured from the perceived curb. | parallel success rate, curb distance error | `parallel_curb` |
| C10 | Garages and tight structures: pillars, narrow aisles, scraping entrances, wrong side of a home garage | Garage scenarios with pillars, narrow aisles and a home garage with a defined charger side. Collision checks use the full footprint including mirrors. | success rate, min clearance | `garage_pillars`, `home_garage` |
| C11 | Charger alignment: stops "a foot short", or port on the wrong side | The goal pose is charge-port aware. For a charger stall the planner picks nose-in or reverse-in so the port faces the charger, and pulls in as deep as the curb allows so the cable reaches. | port-to-charger distance error | `charger_stall`, `home_garage` |

### Perception

| # | Complaint | Engineering response | Metric | Scenario(s) |
|---|---|---|---|---|
| C6 | Fails without painted lines | Spots are inferred from gaps between perceived parked vehicles along the dominant parking orientation. | success rate on unmarked scenarios | `unmarked_lot` |
| C7 | Close-range blindness: posts, bollards, low objects, curbs; objects forgotten once out of view; worse at night | Camera-only sensor model with near-field blind zones and a day/night lighting parameter. A persistent log-odds occupancy grid remembers obstacles after they leave the field of view (object permanence). Safety margins widen when perception confidence drops. | collisions (must be 0), min clearance, curb contact | `night_bollards`, `garage_pillars` |
| C8 | Phantom braking and false stops (shadows, puddles, noise) | Track persistence filtering: an obstacle must be confirmed over N frames before it can cause a hard brake. Real fast-approaching agents still trigger an immediate stop through time-to-collision. | false-stop rate, missed-hazard rate (must be 0) | `phantom_shadows` |
| C12 | Dynamic agents: pedestrians, cars backing out, carts | Tracked agents with constant-velocity prediction plus a "car backing out" intent cue (reverse lights). The car yields, then resumes confidently once the agent clears. | yield events, collisions (0), time lost | `dynamic_agents` |

### Judgment and legality

| # | Complaint | Engineering response | Metric | Scenario(s) |
|---|---|---|---|---|
| C4 | Bad destination/spot choice: far from the entrance, "first available", tight spots between trucks | Destination-aware ranking. The drop-off pin is the scenario's entrance, not the lot centroid. A weighted cost covers walking distance, maneuver difficulty, neighbor clearance and pull-through availability. Weights are user-adjustable and every ranking comes with reasons. | selection matches oracle-best spot (%), walking distance | `destination_entrance` |
| C5 | Parks in restricted or illegal places: accessible, EV-only, expectant-mother, motorcycle, reserved, loading zones, fire lanes, sidewalk ramps, private driveways | A spot and surface classifier accumulates evidence from simulated paint, sign and curb features. Legality is a hard constraint. A restricted spot is used only when the user unlocks it (placard, charging). | violations (must be 0) | `restricted_spots` |
| C15 | Forward/reverse preference ignored; no memory of favorite spots | User preference (reverse-in, nose-in, pull-through, auto) plus per-location preference memory in a JSON file store. An override at a location is remembered and weighted on the next visit. | preference honored (%), repeat-visit match rate | `repeat_visit` |

### Transparency and control

| # | Complaint | Engineering response | Metric | Scenario(s) |
|---|---|---|---|---|
| C13 | Can't see which spot it chose, why, or what it is waiting for; can't easily pick another | Browser HMI showing the chosen spot, ranked alternatives with reasons, the planned path, the current state with its reason ("Yielding to pedestrian"), and live clearance readouts. Click a spot to override; pause, abort and sim-speed controls. | qualitative (README screenshots) plus a Playwright e2e test | `override_spot`, all scenarios in the UI |

## Sources

- Basenor, "FSD intervention data: parking is now the #1 problem": <https://www.basenor.com/blogs/news/fsd-intervention-data-parking-is-now-the-1-problem>
- Electrek (2025-01-07), Actually Smart Summon investigation: <https://electrek.co/2025/01/07/tesla-is-under-investigation-for-actually-smart-summon-crashes/>
- NHTSA ODI, PE24033 opening resume (PDF): <https://static.nhtsa.gov/odi/inv/2024/INOA-PE24033-35917.pdf>
- InsideEVs, FSD new version improvements: <https://insideevs.com/news/742973/tesla-fsd-new-version-improvements/>
- Notebookcheck, vision-only Park Assist rollout: <https://www.notebookcheck.net/Tesla-Vision-only-Park-Assist-without-ultrasonic-sensors-rolls-out-to-more-cars-but-performance-varies.703262.0.html>
- Not a Tesla App, feature request: support handicap spaces for FSD parking: <https://www.notateslaapp.com/tesla-feature-requests/182/support-handicap-spaces-for-fsd-parking/>
- Electrek (2026-06-18), FSD parking preferences and voice control: <https://electrek.co/2026/06/18/tesla-fsd-parking-preferences-grok-voice-control/>

[basenor]: https://www.basenor.com/blogs/news/fsd-intervention-data-parking-is-now-the-1-problem
[electrek-ass]: https://electrek.co/2025/01/07/tesla-is-under-investigation-for-actually-smart-summon-crashes/
[nhtsa]: https://static.nhtsa.gov/odi/inv/2024/INOA-PE24033-35917.pdf
[insideevs]: https://insideevs.com/news/742973/tesla-fsd-new-version-improvements/
[notebookcheck]: https://www.notebookcheck.net/Tesla-Vision-only-Park-Assist-without-ultrasonic-sensors-rolls-out-to-more-cars-but-performance-varies.703262.0.html
[notateslaapp]: https://www.notateslaapp.com/tesla-feature-requests/182/support-handicap-spaces-for-fsd-parking/
[electrek-prefs]: https://electrek.co/2026/06/18/tesla-fsd-parking-preferences-grok-voice-control/
