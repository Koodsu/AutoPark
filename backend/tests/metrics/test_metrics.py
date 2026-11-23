from pathlib import Path

import numpy as np

from parkpilot.metrics.bench import run_benchmark, run_one
from parkpilot.metrics.report import headline_rows, load_results, save_results, write_report
from parkpilot.metrics.run_metrics import gear_changes, spot_is_legal
from parkpilot.scenarios.loader import load_scenario
from parkpilot.scenarios.schema import build_world


def test_gear_change_counting_ignores_jitter() -> None:
    v = np.array([0.0, 0.5, 1.0, 0.02, -0.01, 0.0, -0.4, -0.2, 0.0, 0.3])
    assert gear_changes(v) == 2
    assert gear_changes(np.zeros(5)) == 0


def test_legality_truth() -> None:
    sc = load_scenario("restricted_spots")
    world = build_world(sc)
    legal = {s.id for s in world.spots if spot_is_legal(s, world, sc.user)}
    assert "A3" not in legal and "A8" not in legal  # accessible, motorcycle
    assert "A11" not in legal  # loading zone
    assert {"A10", "A12", "A14", "B9"} <= legal  # neighbours of the zone stay legal


def test_run_one_parkpilot_charger() -> None:
    m = run_one("charger_stall", "parkpilot", 0)
    assert m.parked and m.collisions == 0 and m.violations == 0
    assert m.port_error is not None and m.port_error < 1.2
    assert m.oracle_match is True
    assert m.time_to_park is not None and m.mean_speed > 0.5
    assert m.final_lateral_error is not None and m.final_lateral_error < 0.2


def test_memory_protocol_repeat_visit() -> None:
    pp = run_one("repeat_visit", "parkpilot", 0)
    assert pp.repeat_match is True and pp.preference_honored is True
    assert pp.parked_spot == "A3"


def test_report_generation(tmp_path: Path) -> None:
    results = run_benchmark(["home_garage"], seeds=1, workers=1)
    assert {r.system for r in results} == {"parkpilot", "baseline"}
    path = tmp_path / "results.json"
    save_results(results, path)
    again = load_results(path)
    assert again == results
    md = tmp_path / "docs" / "BENCHMARK.md"
    write_report(results, md, tmp_path / "docs" / "img", 1)
    text = md.read_text()
    assert "| Metric | Baseline | ParkPilot |" in text
    assert (tmp_path / "docs" / "img" / "bench_gear_changes.png").is_file()
    rows = dict((label, (b, p)) for label, b, p in headline_rows(results))
    assert rows["Collisions (total)"][1] == "0"
