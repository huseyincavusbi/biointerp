"""Bio-Interp command line: run experiments and grids from YAML configs."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .runner import run, run_grid


def _parse_set(items):
    overrides = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if not key:
            raise SystemExit(f"invalid --set value: {item!r}")
        overrides[key] = value
    return overrides


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="biointerp", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run one experiment")
    run_p.add_argument("experiment", help="experiment name (conf/experiment/<name>.yaml)")
    run_p.add_argument("--model", default=None, help="model key (conf/model/<key>.yaml)")
    run_p.add_argument("--dataset", default=None, help="dataset key (conf/dataset/<key>.yaml)")
    run_p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an experiment argument",
    )
    run_p.add_argument("--dry-run", action="store_true", help="print the command without running")

    grid_p = sub.add_parser("grid", help="run a grid YAML")
    grid_p.add_argument("grid", help="path to a conf/grids/*.yaml file")
    grid_p.add_argument("--dry-run", action="store_true", help="print commands without running")

    list_p = sub.add_parser("list", help="list known experiments, models or datasets")
    list_p.add_argument("kind", choices=["experiments", "models", "datasets"])

    args = parser.parse_args(argv)

    if args.command == "run":
        result = run(
            experiment=args.experiment,
            model=args.model,
            dataset=args.dataset,
            overrides=_parse_set(args.set),
            dry_run=args.dry_run,
        )
        return int(result.get("rc", 0))

    if args.command == "grid":
        results = run_grid(args.grid, dry_run=args.dry_run)
        return int(any(r.get("rc", 0) for r in results))

    if args.command == "list":
        from .runner import CONF_DIR

        if args.kind == "experiments":
            names = sorted(p.stem for p in (CONF_DIR / "experiment").glob("*.yaml"))
        elif args.kind == "models":
            names = sorted(p.stem for p in (CONF_DIR / "model").glob("*.yaml"))
        else:
            names = sorted(p.stem for p in (CONF_DIR / "dataset").glob("*.yaml"))
        print("\n".join(names))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
