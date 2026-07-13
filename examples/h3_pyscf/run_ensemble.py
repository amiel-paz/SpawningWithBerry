"""Generate the four deterministic H3 observable datasets used in regression work."""

from dataclasses import replace
from pathlib import Path

from aims_berry import load_config, run
from aims_berry.analysis import RunDataset, analyze_run


HERE = Path(__file__).resolve().parent
SEEDS = (1234, 2718, 3141, 5772)


def main() -> None:
    base = load_config(HERE / "aims.in")
    histories = []
    for seed in SEEDS:
        directory = HERE / f"run-seed-{seed}"
        result = run(replace(base, random_seed=seed, run_directory=directory))
        histories.append(result.history)
        analyze_run(result.history, directory / "analysis", base.observables)
    individual, mean = RunDataset.ensemble_populations(histories, state=1)
    dataset = RunDataset(histories[0])
    dataset.export_csv(HERE / "ensemble_populations.csv", individual, tuple(["time_au"] + [f"seed_{seed}" for seed in SEEDS]))
    dataset.export_csv(HERE / "mean_population.csv", mean, ("time_au", "mean_state_1"))


if __name__ == "__main__":
    main()
