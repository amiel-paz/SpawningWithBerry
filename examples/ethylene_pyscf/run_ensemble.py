"""Run the deterministic ethylene corroboration ensemble sequentially."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from aims_berry import load_config, run
from aims_berry.simulation import restart_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("production.in"))
    parser.add_argument("--count", type=int, default=13)
    parser.add_argument("--first-seed", type=int, default=87062)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    base = load_config(args.input)
    ensemble_directory = base.run_directory.with_name("run-ensemble")
    for seed in range(args.first_seed, args.first_seed + args.count):
        member_directory = ensemble_directory / f"seed-{seed}"
        config = dataclasses.replace(
            base,
            random_seed=seed,
            run_directory=member_directory,
        )
        checkpoint = member_directory / "checkpoint" / "current"
        print(f"starting seed {seed}: {member_directory}", flush=True)
        if (checkpoint / "checkpoint.json").is_file():
            result = restart_from_checkpoint(checkpoint)
        else:
            if member_directory.exists() and any(member_directory.iterdir()):
                raise RuntimeError(
                    f"refusing to overwrite nonempty member without checkpoint: {member_directory}"
                )
            result = run(config)
        if result.state.step != config.nsteps:
            raise RuntimeError(
                f"seed {seed} stopped at step {result.state.step}/{config.nsteps}"
            )
        print(f"completed seed {seed}: {result.history}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
