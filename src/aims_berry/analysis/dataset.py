"""Analysis datasets and paper-style observable plots."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from ..config import AU_TIME_PER_FS, BOHR_PER_ANGSTROM


HARTREE_TO_EV = 27.211386245988


def bond(geometry: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(geometry[i] - geometry[j]))


def angle(geometry: np.ndarray, i: int, j: int, k: int) -> float:
    a, b = geometry[i] - geometry[j], geometry[k] - geometry[j]
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def dihedral(geometry: np.ndarray, i: int, j: int, k: int, last: int) -> float:
    b0 = geometry[j] - geometry[i]
    b1 = geometry[k] - geometry[j]
    b2 = geometry[last] - geometry[k]
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


class RunDataset:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def steps(self):
        with h5py.File(self.path, "r") as handle:
            committed = int(handle["steps"].attrs.get("committed_through", 2**63 - 1))
            for name in sorted(handle["steps"]):
                if int(name) > committed:
                    continue
                group = handle["steps"][name]
                frame = {
                    "step": int(name), "time": float(group.attrs["time"]),
                    "positions": group["positions"][...], "states": group["states"][...],
                    "momenta": group["momenta"][...], "amplitudes": group["amplitudes"][...],
                    "labels": [x.decode() if isinstance(x, bytes) else str(x) for x in group["labels"][...]],
                    "energies": group["energies"][...], "populations": group["populations"][...],
                }
                for key in (
                    "classical_kinetic_energy", "classical_potential_energy",
                    "classical_total_energy", "projected_couplings",
                    "retained_overlap_eigenvalues",
                ):
                    if key in group:
                        frame[key] = group[key][...]
                for key in (
                    "metric_norm", "quantum_energy", "quantum_substeps",
                    "raw_norm_before", "raw_norm_after",
                    "quantum_convergence_error", "metric_compatibility_residual",
                    "hamiltonian_hermiticity_residual", "state_population_sum",
                    "regularization_threshold",
                    "nuclear_time_step", "total_tbf_pairs", "active_tbf_pairs",
                    "screened_tbf_pairs", "electronic_calls", "electronic_seconds",
                    "tbf_electronic_calls", "centroid_electronic_calls",
                    "scf_seconds", "casscf_seconds", "gradient_seconds", "nac_seconds",
                    "gradient_evaluations", "nac_evaluations",
                    "nac_pairs_screened_by_gap", "centroid_cache_hits",
                    "replay_seconds", "checkpoint_count", "checkpoint_seconds",
                    "peak_rss_bytes", "cpu_seconds",
                ):
                    if key in group.attrs:
                        frame[key] = group.attrs[key]
                yield frame

    def select(self, *, states: tuple[int, ...] | None = None, labels: tuple[str, ...] | None = None):
        """Yield history frames restricted to selected TBF states and/or labels."""
        for step in self.steps():
            mask = np.ones(len(step["states"]), dtype=bool)
            if states is not None:
                mask &= np.isin(step["states"], states)
            if labels is not None:
                mask &= np.isin(step["labels"], labels)
            selected = dict(step)
            for key in ("positions", "momenta", "states", "energies", "amplitudes"):
                selected[key] = np.asarray(step[key])[mask]
            selected["labels"] = [label for label, keep in zip(step["labels"], mask) if keep]
            yield selected

    def classical_weighted_coordinate(self, kind: str, atoms: tuple[int, ...]) -> np.ndarray:
        """Return the normalized |amplitude|^2-weighted classical coordinate."""
        function = {"bond": bond, "angle": angle, "dihedral": dihedral}[kind]
        values = []
        for step in self.steps():
            weights = np.abs(step["amplitudes"]) ** 2
            weights /= weights.sum() if weights.sum() else 1.0
            coordinate = sum(
                weight * function(geometry, *atoms)
                for weight, geometry in zip(weights, step["positions"])
            )
            values.append((step["time"], coordinate))
        return np.asarray(values)

    def coordinate(self, kind: str, atoms: tuple[int, ...]) -> dict[str, np.ndarray]:
        function = {"bond": bond, "angle": angle, "dihedral": dihedral}[kind]
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for step in self.steps():
            for label, geometry in zip(step["labels"], step["positions"]):
                series[label].append((step["time"], function(geometry, *atoms)))
        return {label: np.asarray(values) for label, values in series.items()}

    def energy_gap(self, lower: int = 0, upper: int = 1, trajectory_label: str | None = None) -> np.ndarray:
        values = []
        for step in self.steps():
            index = 0 if trajectory_label is None else step["labels"].index(trajectory_label)
            energies = step["energies"][index]
            values.append((step["time"], energies[upper] - energies[lower]))
        return np.asarray(values)

    def populations(self) -> np.ndarray:
        return np.asarray([(step["time"], *step["populations"]) for step in self.steps()])

    def diagnostics(self) -> np.ndarray:
        """Time, norm, substeps, errors, gap, coupling, TBF count, and energy."""
        rows = []
        for step in self.steps():
            gap = (
                float(np.nanmin(np.abs(step["energies"][:, 1] - step["energies"][:, 0])))
                if step["energies"].shape[1] > 1 else np.nan
            )
            couplings = step.get("projected_couplings")
            maximum_coupling = (
                float(np.nanmax(np.abs(couplings)))
                if couplings is not None and np.any(np.isfinite(couplings)) else np.nan
            )
            rows.append((
                step["time"], step.get("metric_norm", np.nan),
                step.get("raw_norm_before", np.nan), step.get("raw_norm_after", np.nan),
                step.get("quantum_substeps", np.nan),
                step.get("quantum_convergence_error", np.nan),
                step.get("metric_compatibility_residual", np.nan), gap,
                maximum_coupling, len(step["states"]),
                step.get("quantum_energy", np.nan),
            ))
        return np.asarray(rows)

    def performance_diagnostics(self) -> np.ndarray:
        """Return cumulative electronic work, sparsity, timing, and memory telemetry."""
        fields = (
            "nuclear_time_step", "total_tbf_pairs", "active_tbf_pairs",
            "screened_tbf_pairs", "electronic_calls", "electronic_seconds",
            "tbf_electronic_calls", "centroid_electronic_calls",
            "scf_seconds", "casscf_seconds", "gradient_seconds", "nac_seconds",
            "gradient_evaluations", "nac_evaluations",
            "nac_pairs_screened_by_gap", "centroid_cache_hits",
            "replay_seconds", "checkpoint_count", "checkpoint_seconds",
            "peak_rss_bytes", "cpu_seconds",
        )
        return np.asarray([
            (step["time"], *(step.get(field, np.nan) for field in fields))
            for step in self.steps()
        ])

    def events(self) -> list[dict]:
        """Return the cumulative event log from the last committed frame."""
        with h5py.File(self.path, "r") as handle:
            steps = handle["steps"]
            committed = int(steps.attrs.get("committed_through", -1))
            names = [name for name in steps if int(name) <= committed]
            if not names:
                return []
            payload = steps[max(names)].attrs.get("events_json", "[]")
            return json.loads(payload)

    def summary(self) -> dict:
        """Return numerical gates and scientific headline values for one member."""
        frames = list(self.steps())
        if not frames:
            raise ValueError(f"no committed frames in {self.path}")
        populations = self.populations()
        gaps = self.energy_gap()
        diagnostics = self.diagnostics()
        classical = self.classical_energies()
        classical_drifts = [
            float(np.max(np.abs(values[:, 3] - values[0, 3])))
            for values in classical.values() if len(values)
        ]
        spawns = [event for event in self.events() if event.get("kind") == "spawn"]
        return {
            "history": str(self.path),
            "final_time_au": float(frames[-1]["time"]),
            "final_time_fs": float(frames[-1]["time"] / AU_TIME_PER_FS),
            "initial_excitation_ev": float(
                (frames[0]["energies"][0, 1] - frames[0]["energies"][0, 0])
                * HARTREE_TO_EV
            ) if frames[0]["energies"].shape[1] > 1 else None,
            "final_state_populations": populations[-1, 1:].tolist(),
            "minimum_gap_hartree": float(np.nanmin(gaps[:, 1])),
            "minimum_gap_ev": float(np.nanmin(gaps[:, 1]) * HARTREE_TO_EV),
            "minimum_gap_time_fs": float(gaps[np.nanargmin(gaps[:, 1]), 0] / AU_TIME_PER_FS),
            "accepted_spawn_count": len(spawns),
            "maximum_metric_norm_error": float(
                np.nanmax(np.abs(diagnostics[:, 1] - diagnostics[0, 1]))
            ),
            "maximum_population_sum_error": float(
                np.max(np.abs(np.sum(populations[:, 1:], axis=1) - 1.0))
            ),
            "maximum_quantum_energy_drift_hartree": float(
                np.nanmax(np.abs(diagnostics[:, 10] - diagnostics[0, 10]))
            ),
            "maximum_classical_energy_drift_hartree": (
                max(classical_drifts) if classical_drifts else None
            ),
        }

    def ethylene_spawn_geometries(
        self, *, migration_cutoff_angstrom: float = 1.6
    ) -> list[dict]:
        """Classify accepted spawn geometries with a documented operational rule.

        A short cross-carbon C--H contact is labelled ethylidene-like; remaining
        events are labelled twisted/pyramidalized. The descriptor is intentionally
        exposed so the final report can state the classification convention rather
        than silently claim the paper's unavailable trajectory labels.
        """
        frames = list(self.steps())
        output = []
        for event in self.events():
            if event.get("kind") != "spawn":
                continue
            frame = min(frames, key=lambda item: abs(item["time"] - event["time"]))
            label = event.get("parent")
            index = frame["labels"].index(label) if label in frame["labels"] else 0
            geometry = frame["positions"][index] / BOHR_PER_ANGSTROM
            cross = [
                bond(geometry, 0, 4), bond(geometry, 0, 5),
                bond(geometry, 1, 2), bond(geometry, 1, 3),
            ]
            minimum = float(min(cross))
            output.append({
                "time_au": float(event["time"]),
                "time_fs": float(event["time"] / AU_TIME_PER_FS),
                "parent": label,
                "child": event.get("child"),
                "minimum_cross_carbon_ch_angstrom": minimum,
                "classification": (
                    "ethylidene-like"
                    if minimum < migration_cutoff_angstrom
                    else "twisted-pyramidalized"
                ),
            })
        return output

    def classical_energies(self) -> dict[str, np.ndarray]:
        series: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        for step in self.steps():
            if "classical_total_energy" not in step:
                continue
            for index, label in enumerate(step["labels"]):
                series[label].append((
                    step["time"], step["classical_kinetic_energy"][index],
                    step["classical_potential_energy"][index],
                    step["classical_total_energy"][index],
                ))
        return {label: np.asarray(values) for label, values in series.items()}

    @staticmethod
    def ensemble_populations(paths: list[str | Path], state: int = 1) -> tuple[np.ndarray, np.ndarray]:
        runs = [RunDataset(path).populations() for path in paths]
        start = max(run[:, 0].min() for run in runs)
        stop = min(run[:, 0].max() for run in runs)
        count = min(len(run) for run in runs)
        times = np.linspace(start, stop, count)
        values = np.asarray([np.interp(times, run[:, 0], run[:, state + 1]) for run in runs])
        return np.column_stack((times, values.T)), np.column_stack((times, values.mean(axis=0)))

    def export_csv(self, output: str | Path, rows: np.ndarray, headers: tuple[str, ...]) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows.tolist())
        return output


def analyze_run(path: str | Path, output_directory: str | Path, observables=()) -> list[Path]:
    with h5py.File(path, "r") as handle:
        if "steps" not in handle and "sim/qm_amplitudes" in handle:
            from .pyspawn import analyze_pyspawn_reference
            return analyze_pyspawn_reference(path, output_directory)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires the 'analysis' optional dependency") from exc
    dataset = RunDataset(path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    products: list[Path] = []
    for observable in observables:
        series = dataset.coordinate(observable.kind, observable.atoms)
        figure, axis = plt.subplots()
        for label, values in series.items():
            axis.plot(values[:, 0], values[:, 1], label=label)
        axis.set(xlabel="Time (a.u.)", ylabel=observable.kind.title())
        axis.legend()
        target = output / f"{observable.name}.png"
        figure.savefig(target, dpi=160, bbox_inches="tight")
        plt.close(figure)
        products.append(target)
        for label, values in series.items():
            csv_target = output / f"{observable.name}-{label}.csv"
            dataset.export_csv(
                csv_target, values, ("time_au", observable.kind)
            )
            products.append(csv_target)
    first = next(dataset.steps())
    if first["energies"].shape[1] >= 2:
        gap = dataset.energy_gap()
        figure, axis = plt.subplots()
        axis.plot(gap[:, 0], gap[:, 1])
        axis.set(xlabel="Time (a.u.)", ylabel="Energy gap (Eh)")
        target = output / "energy_gap.png"
        figure.savefig(target, dpi=160, bbox_inches="tight")
        plt.close(figure)
        products.append(target)
        dataset.export_csv(output / "energy_gap.csv", gap, ("time_au", "gap_hartree"))
    populations = dataset.populations()
    figure, axis = plt.subplots()
    for state in range(populations.shape[1] - 1):
        axis.plot(populations[:, 0], populations[:, state + 1], label=f"S{state}")
    axis.set(xlabel="Time (a.u.)", ylabel="Population", ylim=(0, 1.05))
    axis.legend()
    target = output / "state_populations.png"
    figure.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(figure)
    products.append(target)
    dataset.export_csv(output / "populations.csv", populations, tuple(["time_au"] + [f"state_{i}" for i in range(populations.shape[1] - 1)]))
    diagnostics = dataset.diagnostics()
    diagnostic_headers = (
        "time_au", "metric_norm", "raw_norm_before", "raw_norm_after",
        "quantum_substeps", "coefficient_error", "metric_residual",
        "minimum_gap_hartree", "maximum_projected_coupling", "tbf_count",
        "quantum_energy_hartree",
    )
    diagnostic_csv = dataset.export_csv(
        output / "propagation_diagnostics.csv", diagnostics, diagnostic_headers
    )
    products.append(diagnostic_csv)
    performance = dataset.performance_diagnostics()
    performance_headers = (
        "time_au", "nuclear_time_step_au", "total_tbf_pairs",
        "active_tbf_pairs", "screened_tbf_pairs", "electronic_calls",
        "electronic_seconds", "tbf_electronic_calls", "centroid_electronic_calls",
        "scf_seconds", "casscf_seconds", "gradient_seconds", "nac_seconds",
        "gradient_evaluations", "nac_evaluations",
        "nac_pairs_screened_by_gap", "centroid_cache_hits", "replay_seconds",
        "checkpoint_count", "checkpoint_seconds", "peak_rss_bytes", "cpu_seconds",
    )
    products.append(dataset.export_csv(
        output / "performance_diagnostics.csv", performance, performance_headers
    ))
    summary = dataset.summary()
    summary["ethylene_spawn_geometries"] = (
        dataset.ethylene_spawn_geometries()
        if first["positions"].shape[1] == 6 else []
    )
    summary_path = output / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    products.append(summary_path)
    if diagnostics.size:
        panels = (
            (1, "Metric norm"), (4, "Quantum substeps"),
            (8, "Projected coupling"), (9, "TBF count"),
        )
        figure, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(7, 8))
        for axis, (column, label) in zip(axes, panels):
            axis.plot(diagnostics[:, 0], diagnostics[:, column])
            axis.set_ylabel(label)
        axes[-1].set_xlabel("Time (a.u.)")
        target = output / "propagation_diagnostics.png"
        figure.savefig(target, dpi=160, bbox_inches="tight")
        plt.close(figure)
        products.append(target)
    for label, values in dataset.classical_energies().items():
        products.append(dataset.export_csv(
            output / f"classical_energy-{label}.csv",
            values,
            ("time_au", "kinetic_hartree", "potential_hartree", "total_hartree"),
        ))
    return products
