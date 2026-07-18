"""Run a monitored ethylene ensemble without oversubscribing electronic workers."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from aims_berry import load_config, run
from aims_berry.config import config_from_dict

PHYSICAL_CORES = 12
ELECTRONIC_THREADS = 2
ELECTRONIC_SLOTS = PHYSICAL_CORES // ELECTRONIC_THREADS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("production.in"))
    parser.add_argument("--count", type=int, default=13)
    parser.add_argument("--first-seed", type=int, default=87062)
    parser.add_argument("--parallel-members", type=int, default=6)
    parser.add_argument("--production-count", type=int, default=13)
    parser.add_argument("--simulation-time", type=float)
    parser.add_argument("--time-step", type=float)
    parser.add_argument("--pair-overlap-threshold", type=float)
    parser.add_argument("--ensemble-name", default="run-ensemble")
    parser.add_argument("--memory-limit-gb", type=float, default=32.0)
    parser.add_argument("--runtime-limit-hours", type=float, default=72.0)
    parser.add_argument("--max-projected-hours", type=float, default=72.0)
    parser.add_argument("--member-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--member-workers", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def _provider_options(config, workers: int) -> tuple[tuple[str, object], ...]:
    options = config.provider_option_dict()
    options["electronic_workers"] = int(workers)
    options["max_memory"] = 2500
    return tuple(options.items())


def _apply_completion_policies(checkpoint_config, input_config):
    """Apply explicitly mutable run policies when resuming scientific state.

    Nuclear/electronic state, RNG, queues, and all numerical controls still come
    from the checkpoint.  These two policies only decide whether already-recorded
    diagnostics stop the run or are retained as warnings.
    """
    options = checkpoint_config.provider_option_dict()
    input_options = input_config.provider_option_dict()
    if "continuity_policy" in input_options:
        options["continuity_policy"] = input_options["continuity_policy"]
    return dataclasses.replace(
        checkpoint_config,
        provider_options=tuple(options.items()),
        classical_energy_policy=input_config.classical_energy_policy,
        quantum_energy_policy=input_config.quantum_energy_policy,
    )


def _run_member(args: argparse.Namespace) -> int:
    base = load_config(args.input)
    ensemble_directory = base.run_directory.with_name(args.ensemble_name)
    member_directory = ensemble_directory / f"seed-{args.member_seed}"
    checkpoint = member_directory / "checkpoint" / "current"
    if (checkpoint / "checkpoint.json").is_file():
        metadata = json.loads((checkpoint / "checkpoint.json").read_text())
        config = config_from_dict(metadata["config"])
        config = _apply_completion_policies(config, base)
        config = dataclasses.replace(config, run_directory=member_directory)
        restart = True
    else:
        existing_products = (
            [path for path in member_directory.iterdir() if path.name != "runner.log"]
            if member_directory.exists() else []
        )
        if existing_products:
            raise RuntimeError(
                f"refusing to overwrite nonempty member without checkpoint: {member_directory}"
            )
        config = dataclasses.replace(
            base,
            random_seed=args.member_seed,
            run_directory=member_directory,
        )
        restart = False
    config = dataclasses.replace(
        config,
        provider_options=_provider_options(config, args.member_workers),
        time_step=config.time_step if args.time_step is None else args.time_step,
        pair_overlap_threshold=(
            config.pair_overlap_threshold
            if args.pair_overlap_threshold is None
            else args.pair_overlap_threshold
        ),
        simulation_time=(
            config.simulation_time if args.simulation_time is None else args.simulation_time
        ),
    )
    print(
        f"starting seed {args.member_seed}: {member_directory} "
        f"workers={args.member_workers}",
        flush=True,
    )
    result = run(config, restart=restart)
    if result.state.quantum_time < config.simulation_time - 1.0e-10:
        raise RuntimeError(
            f"seed {args.member_seed} stopped at "
            f"{result.state.quantum_time}/{config.simulation_time} au"
        )
    print(f"completed seed {args.member_seed}: {result.history}", flush=True)
    return 0


def _process_tree_rss(pids: set[int]) -> int:
    listing = subprocess.run(
        ["ps", "-axo", "ppid=,pid=,rss="], check=True, capture_output=True, text=True
    ).stdout
    rows = [tuple(map(int, line.split())) for line in listing.splitlines() if line.split()]
    descendants = set(pids)
    changed = True
    while changed:
        changed = False
        for parent, child, _rss in rows:
            if parent in descendants and child not in descendants:
                descendants.add(child)
                changed = True
    return 1024 * sum(rss for _parent, pid, rss in rows if pid in descendants)


def _swap_used_bytes() -> int:
    try:
        output = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], check=True, capture_output=True, text=True
        ).stdout
        match = re.search(r"used = ([0-9.]+)([MG])", output)
        if match is None:
            return 0
        scale = 1024**2 if match.group(2) == "M" else 1024**3
        return int(float(match.group(1)) * scale)
    except (OSError, subprocess.SubprocessError):
        return 0


def _member_progress(directory: Path) -> dict[str, float]:
    checkpoint = directory / "checkpoint" / "current" / "checkpoint.json"
    if not checkpoint.is_file():
        return {"time_au": 0.0, "electronic_calls": 0.0}
    metadata = json.loads(checkpoint.read_text())
    metrics = metadata.get("runtime", {}).get("runtime_metrics", {})
    output = {
        "time_au": float(metadata.get("quantum_time", 0.0)),
    }
    for key in (
        "electronic_calls", "electronic_seconds", "tbf_electronic_calls",
        "centroid_electronic_calls", "scf_seconds", "casscf_seconds",
        "gradient_seconds", "nac_seconds", "gradient_evaluations",
        "nac_evaluations", "nac_pairs_screened_by_gap", "centroid_cache_hits",
        "replay_seconds", "checkpoint_count", "checkpoint_seconds",
    ):
        output[key] = float(metrics.get(key, 0.0))
    return output


def _write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _stop_processes(processes: dict[int, tuple[subprocess.Popen, object]]) -> None:
    for process, _stream in processes.values():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(
        process.poll() is None for process, _stream in processes.values()
    ):
        time.sleep(0.2)
    for process, _stream in processes.values():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def _run_batch(
    args: argparse.Namespace,
    seeds: list[int],
    workers: int,
    prior_failures: list[dict] | None = None,
) -> tuple[list[float], list[dict]]:
    base = load_config(args.input)
    ensemble_directory = base.run_directory.with_name(args.ensemble_name)
    ensemble_directory.mkdir(parents=True, exist_ok=True)
    processes: dict[int, tuple[subprocess.Popen, object]] = {}
    launched: dict[int, float] = {}
    durations: list[float] = []
    completed_seeds: set[int] = set()
    member_failures: list[dict] = []
    prior_failures = [] if prior_failures is None else list(prior_failures)
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": str(ELECTRONIC_THREADS),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    for seed in seeds:
        member = ensemble_directory / f"seed-{seed}"
        member.mkdir(parents=True, exist_ok=True)
        failure_path = member / "failure.json"
        if failure_path.is_file():
            record = json.loads(failure_path.read_text())
            member_failures.append(record)
            print(
                f"skipping quarantined seed {seed}; remove {failure_path} "
                "after post-mortem repair to retry it",
                flush=True,
            )
            continue
        stream = (member / "runner.log").open("a")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--input", str(args.input.resolve()),
            "--ensemble-name", args.ensemble_name,
            "--member-seed", str(seed),
            "--member-workers", str(workers),
        ]
        if args.simulation_time is not None:
            command.extend(["--simulation-time", str(args.simulation_time)])
        if args.time_step is not None:
            command.extend(["--time-step", str(args.time_step)])
        if args.pair_overlap_threshold is not None:
            command.extend([
                "--pair-overlap-threshold", str(args.pair_overlap_threshold)
            ])
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes[seed] = (process, stream)
        launched[seed] = time.monotonic()

    initial_swap = _swap_used_bytes()
    swap_strikes = 0
    batch_started = time.monotonic()
    peak_rss = 0
    peak_swap_growth = 0
    try:
        while processes:
            failed = []
            completed = []
            for seed, (process, stream) in processes.items():
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                elapsed = time.monotonic() - launched[seed]
                (failed if code else completed).append((seed, code, elapsed))
            for seed, code, elapsed in completed:
                durations.append(elapsed)
                completed_seeds.add(seed)
                del processes[seed]
            for seed, code, elapsed in failed:
                record = {
                    "seed": seed,
                    "exit_code": code,
                    "elapsed_seconds": elapsed,
                    "progress": _member_progress(
                        ensemble_directory / f"seed-{seed}"
                    ),
                }
                member_failures.append(record)
                _write_status(
                    ensemble_directory / f"seed-{seed}" / "failure.json",
                    record,
                )
                del processes[seed]
                print(
                    f"quarantined failed seed {seed} with status {code}; "
                    "continuing ensemble",
                    flush=True,
                )

            pids = {process.pid for process, _stream in processes.values()}
            rss = _process_tree_rss(pids) if pids else 0
            swap_growth = max(0, _swap_used_bytes() - initial_swap)
            peak_rss = max(peak_rss, rss)
            peak_swap_growth = max(peak_swap_growth, swap_growth)
            swap_strikes = swap_strikes + 1 if swap_growth > 512 * 1024**2 else 0
            progress = {
                str(seed): _member_progress(ensemble_directory / f"seed-{seed}")
                for seed in seeds
            }
            elapsed = time.monotonic() - batch_started
            rates = [
                elapsed / item["time_au"]
                for item in progress.values() if item["time_au"] > 0
            ]
            projected = None
            if rates:
                projected = (
                    max(rates) * base.simulation_time * args.production_count
                    / args.parallel_members / 3600.0
                )
            _write_status(ensemble_directory / "ensemble-status.json", {
                "active_seeds": sorted(processes),
                "batch_seeds": seeds,
                "completed_seeds": sorted(completed_seeds),
                "failed_members": prior_failures + member_failures,
                "rss_bytes": rss,
                "peak_rss_bytes": peak_rss,
                "swap_growth_bytes": swap_growth,
                "peak_swap_growth_bytes": peak_swap_growth,
                "elapsed_seconds": elapsed,
                "projected_ensemble_hours": projected,
                "progress": progress,
            })
            if rss > args.memory_limit_gb * 1024**3:
                raise RuntimeError(f"ensemble RSS {rss / 1024**3:.2f} GB exceeds limit")
            if swap_strikes >= 2:
                raise RuntimeError("sustained swap growth exceeded 512 MB")
            if elapsed > args.runtime_limit_hours * 3600:
                raise RuntimeError("ensemble runtime limit exceeded")
            time.sleep(2.0)
    except BaseException:
        _stop_processes(processes)
        raise
    finally:
        for _process, stream in processes.values():
            if not stream.closed:
                stream.close()
    return durations, member_failures


def main() -> int:
    args = parse_args()
    if args.member_seed is not None:
        return _run_member(args)
    if args.count < 1 or not 1 <= args.parallel_members <= ELECTRONIC_SLOTS:
        raise ValueError("count must be positive and parallel-members must be in [1, 6]")
    if args.production_count < 1:
        raise ValueError("production-count must be positive")
    seeds = list(range(args.first_seed, args.first_seed + args.count))
    durations: list[float] = []
    failures: list[dict] = []
    for offset in range(0, len(seeds), args.parallel_members):
        batch = seeds[offset:offset + args.parallel_members]
        workers = max(1, ELECTRONIC_SLOTS // len(batch))
        batch_durations, batch_failures = _run_batch(
            args, batch, workers, failures
        )
        durations.extend(batch_durations)
        failures.extend(batch_failures)
    if args.simulation_time and durations:
        base = load_config(args.input)
        extrapolated = (
            max(durations) / args.simulation_time * base.simulation_time
            * args.production_count / args.parallel_members / 3600.0
        )
        print(f"projected full-ensemble makespan: {extrapolated:.2f} hours", flush=True)
        if extrapolated > args.max_projected_hours:
            raise RuntimeError(
                f"projected makespan {extrapolated:.2f} h exceeds "
                f"{args.max_projected_hours:.2f} h gate"
            )
    if failures:
        base = load_config(args.input)
        ensemble_directory = base.run_directory.with_name(args.ensemble_name)
        _write_status(ensemble_directory / "ensemble-failures.json", {
            "failed_members": failures,
            "failed_seeds": [item["seed"] for item in failures],
        })
        print(
            "ensemble completed with quarantined seed failures: "
            + ", ".join(str(item["seed"]) for item in failures),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
