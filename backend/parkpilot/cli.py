"""Command-line entry point: ``parkpilot serve | bench | list | export-schema``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cmd_list(_: argparse.Namespace) -> int:
    from parkpilot.scenarios.loader import list_scenarios

    for sc in list_scenarios():
        print(f"{sc.id:24s} {','.join(sc.complaints):10s} {sc.title}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from parkpilot.metrics.bench import run_benchmark
    from parkpilot.metrics.report import save_results, write_report

    scenarios = args.scenarios.split(",") if args.scenarios else None
    results = run_benchmark(scenarios, args.seeds, workers=args.workers)
    docs = Path(args.docs)
    save_results(results, Path(args.results))
    write_report(results, docs / "BENCHMARK.md", docs / "img", args.seeds)
    bad = [
        r
        for r in results
        if r.system == "parkpilot" and (r.collisions or r.violations or r.missed_hazards)
    ]
    print(f"{len(results)} runs; report written to {docs / 'BENCHMARK.md'}")
    if bad:
        print(f"WARNING: {len(bad)} ParkPilot runs with collisions/violations/missed hazards")
        return 1 if args.strict else 0
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("parkpilot.api.server:app", host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_export_schema(args: argparse.Namespace) -> int:
    from parkpilot.api.protocol import export_schema

    out = Path(args.out)
    out.write_text(export_schema() + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parkpilot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list scenarios").set_defaults(fn=_cmd_list)
    b = sub.add_parser("bench", help="run the A/B benchmark and write docs/BENCHMARK.md")
    b.add_argument("--seeds", type=int, default=5)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--scenarios", default="")
    b.add_argument("--docs", default=str(REPO_ROOT / "docs"))
    b.add_argument("--results", default=str(REPO_ROOT / "docs" / "benchmark_results.json"))
    b.add_argument("--strict", action="store_true", help="fail on any ParkPilot safety issue")
    b.set_defaults(fn=_cmd_bench)
    s = sub.add_parser("serve", help="run the API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_cmd_serve)
    e = sub.add_parser("export-schema", help="write the wire-protocol JSON Schema")
    e.add_argument("--out", default=str(REPO_ROOT / "frontend" / "schema" / "protocol.schema.json"))
    e.set_defaults(fn=_cmd_export_schema)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
