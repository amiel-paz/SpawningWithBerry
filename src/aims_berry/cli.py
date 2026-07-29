"""Command-line interface for validation, dynamics, restart, and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_run
from .config import ConfigError, load_config
from .geometry import masses_and_widths, read_xyz
from .io import export_xyz_history
from .simulation import restart_from_checkpoint, run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aims-berry")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="parse and validate an input file")
    validate.add_argument("input", type=Path)
    execute = commands.add_parser("run", help="start a simulation")
    execute.add_argument("input", type=Path)
    restart = commands.add_parser("restart", help="resume an exact checkpoint")
    restart.add_argument("checkpoint", type=Path)
    restart.add_argument(
        "--simulation-time",
        type=float,
        help="extend the checkpoint endpoint in atomic units",
    )
    analyze = commands.add_parser("analyze", help="export observables and plots")
    analyze.add_argument("history", type=Path, help="simulation.h5 or its run directory")
    analyze.add_argument("--input", type=Path, help="input file defining observable records")
    analyze.add_argument("--output", type=Path, default=Path("analysis"))
    export_xyz = commands.add_parser(
        "export-xyz", help="export committed TBF geometries from simulation.h5"
    )
    export_xyz.add_argument(
        "history", type=Path, help="simulation.h5 or its run directory"
    )
    export_xyz.add_argument("--output", type=Path, default=Path("geometries"))
    return parser


def _validated_config(path: Path):
    config = load_config(path)
    atoms, positions = read_xyz(config.geometry, config.geometry_units)
    masses_and_widths(atoms, config.gaussian_widths, config.nuclear_masses)
    if config.momenta is not None and not config.momenta.is_file():
        raise ConfigError(f"momenta file does not exist: {config.momenta}")
    if config.hessian is not None and not config.hessian.is_file():
        raise ConfigError(f"Hessian file does not exist: {config.hessian}")
    return config, atoms, positions


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        config, atoms, positions = _validated_config(args.input)
        print(json.dumps({
            "valid": True,
            "provider": config.provider,
            "atoms": len(atoms),
            "degrees_of_freedom": int(positions.size),
            "states": config.num_states,
            "steps": config.nsteps,
            "run_directory": str(config.run_directory),
        }, indent=2))
        return 0
    if args.command == "run":
        config, _, _ = _validated_config(args.input)
        result = run(config)
        print(result.history)
        return 0
    if args.command == "restart":
        result = restart_from_checkpoint(
            args.checkpoint,
            simulation_time=args.simulation_time,
        )
        print(result.history)
        return 0
    if args.command == "export-xyz":
        print(export_xyz_history(args.history, args.output))
        return 0
    history = args.history / "simulation.h5" if args.history.is_dir() else args.history
    observables = load_config(args.input).observables if args.input else ()
    for product in analyze_run(history, args.output, observables):
        print(product)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
