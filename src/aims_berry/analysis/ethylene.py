"""Ensemble-level ethylene corroboration summaries."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from ..config import AU_TIME_PER_FS
from .dataset import RunDataset


def _lifetime(times_au: np.ndarray, population: np.ndarray) -> tuple[float, float]:
    times_fs = np.asarray(times_au, dtype=float) / AU_TIME_PER_FS
    values = np.asarray(population, dtype=float)
    mask = np.isfinite(values) & (values > 1.0e-6)
    if np.count_nonzero(mask) < 3 or np.ptp(times_fs[mask]) <= 0:
        return float("nan"), float("nan")

    def decay(time_fs, lifetime_fs):
        return np.exp(-time_fs / lifetime_fs)

    fit, covariance = curve_fit(
        decay,
        times_fs[mask],
        values[mask],
        p0=(110.0,),
        bounds=(1.0e-6, np.inf),
        maxfev=10000,
    )
    uncertainty = float(np.sqrt(max(covariance[0, 0], 0.0)))
    return float(fit[0]), uncertainty


def ethylene_ensemble_report(paths: list[str | Path]) -> dict:
    """Compare a completed ensemble with the 2009 CASSCF targets.

    The lifetime is a one-parameter fit of the ensemble-mean S1 population to
    ``exp(-t/tau)``. Spawn-pathway labels use the explicit cross-carbon C--H
    descriptor implemented by :meth:`RunDataset.ethylene_spawn_geometries`.
    """
    datasets = [RunDataset(path) for path in paths]
    if not datasets:
        raise ValueError("at least one simulation history is required")
    individual, mean = RunDataset.ensemble_populations(
        [dataset.path for dataset in datasets], state=1
    )
    lifetime, lifetime_uncertainty = _lifetime(mean[:, 0], mean[:, 1])
    summaries = [dataset.summary() for dataset in datasets]
    classifications = Counter(
        spawn["classification"]
        for dataset in datasets
        for spawn in dataset.ethylene_spawn_geometries()
    )
    final_s0 = np.asarray([
        summary["final_state_populations"][0] for summary in summaries
    ])
    excitations = np.asarray([summary["initial_excitation_ev"] for summary in summaries])
    gaps = np.asarray([summary["minimum_gap_ev"] for summary in summaries])
    return {
        "member_count": len(datasets),
        "all_members_reach_250_fs": all(
            summary["final_time_fs"] >= 250.0 - 0.1 for summary in summaries
        ),
        "initial_excitation_ev_mean": float(np.mean(excitations)),
        "initial_excitation_ev_std": float(np.std(excitations, ddof=1)) if len(excitations) > 1 else 0.0,
        "reference_initial_excitation_ev": 10.12,
        "s1_lifetime_fs": lifetime,
        "s1_lifetime_fit_uncertainty_fs": lifetime_uncertainty,
        "reference_s1_lifetime_fs": 110.0,
        "reference_s1_lifetime_uncertainty_fs": 6.0,
        "final_s0_population_mean": float(np.mean(final_s0)),
        "final_s0_population_std": float(np.std(final_s0, ddof=1)) if len(final_s0) > 1 else 0.0,
        "reference_final_s0_population": 0.97,
        "accepted_spawn_count": int(sum(
            summary["accepted_spawn_count"] for summary in summaries
        )),
        "minimum_gap_ev": float(np.min(gaps)),
        "member_minimum_gaps_ev": gaps.tolist(),
        "spawn_pathway_counts": dict(classifications),
        "spawn_pathway_classification": (
            "ethylidene-like when the nearest cross-carbon C-H distance is below "
            "1.6 angstrom; twisted-pyramidalized otherwise"
        ),
        "maximum_metric_norm_error": float(max(
            summary["maximum_metric_norm_error"] for summary in summaries
        )),
        "maximum_population_sum_error": float(max(
            summary["maximum_population_sum_error"] for summary in summaries
        )),
        "maximum_quantum_energy_drift_hartree": float(max(
            summary["maximum_quantum_energy_drift_hartree"] for summary in summaries
        )),
        "maximum_classical_energy_drift_hartree": float(max(
            summary["maximum_classical_energy_drift_hartree"] or 0.0
            for summary in summaries
        )),
        "ensemble_population_points": int(len(mean)),
        "individual_population_shape": list(individual.shape),
    }


def write_ethylene_ensemble_report(
    paths: list[str | Path], output: str | Path
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(ethylene_ensemble_report(paths), indent=2, sort_keys=True) + "\n"
    )
    return target
