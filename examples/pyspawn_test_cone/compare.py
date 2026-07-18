"""Compare native-PySpawn/full-diagonal and SpawnWithBerry/Cayley histories."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
PYSPAWN = ROOT / "run-pyspawn-harmonized-fulldiag" / "sim.hdf5"
PYSPAWN_PRISTINE = ROOT / "run-pyspawn-pristine-fulldiag" / "sim.hdf5"
SPAWN_WITH_BERRY = ROOT / "run-spawnwithberry" / "simulation.h5"
ENERGY_SHIFT = -5.18


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _pyspawn_series(path: Path) -> dict:
    with h5py.File(path) as handle:
        sim = handle["sim"]
        time = sim["quantum_time"][:, 0]
        count = sim["num_traj_qm"][:, 0].astype(int)
        width = sim["qm_amplitudes"].shape[1]
        state_populations = np.zeros((len(time), 2))
        amplitude_magnitudes = np.full((len(time), width), np.nan)
        tbf_populations = np.full((len(time), width), np.nan)
        norms = np.zeros(len(time))
        qenergy = np.full(len(time), np.nan)
        labels: list[list[str]] = []
        states: list[list[int]] = []
        for frame, ntraj in enumerate(count):
            frame_labels = _text(sim["labels_this_step"][frame]).split(",")
            frame_states = [int(value) for value in _text(
                sim["istates_this_step"][frame]
            ).split(",")]
            labels.append(frame_labels)
            states.append(frame_states)
            coefficients = sim["qm_amplitudes"][frame, :ntraj]
            overlap = sim["S"][frame, :ntraj * ntraj].reshape(ntraj, ntraj)
            hamiltonian = sim["H"][frame, :ntraj * ntraj].reshape(ntraj, ntraj)
            projected = np.real(np.conjugate(coefficients) * (overlap @ coefficients))
            amplitude_magnitudes[frame, :ntraj] = np.abs(coefficients)
            tbf_populations[frame, :ntraj] = projected
            for index, state in enumerate(frame_states):
                state_populations[frame, state] += projected[index]
            norms[frame] = float(np.real(np.vdot(coefficients, overlap @ coefficients)))
            if abs(norms[frame]) > 1.0e-14:
                physical_h = hamiltonian - ENERGY_SHIFT * overlap
                qenergy[frame] = float(np.real(
                    np.vdot(coefficients, physical_h @ coefficients) / norms[frame]
                ))
        return {
            "time": time,
            "count": count,
            "populations": state_populations,
            "amplitude_magnitudes": amplitude_magnitudes,
            "tbf_populations": tbf_populations,
            "norm": norms,
            "quantum_energy": qenergy,
            "labels": labels,
            "states": states,
            "matrix_width": width,
        }


def _swb_series(path: Path) -> dict:
    with h5py.File(path) as handle:
        steps = handle["steps"]
        names = sorted(steps, key=int)
        time = np.asarray([steps[name].attrs["time"] for name in names])
        count = np.asarray([len(steps[name]["states"]) for name in names], dtype=int)
        populations = np.asarray([steps[name]["populations"][:] for name in names])
        norm = np.asarray([steps[name].attrs["metric_norm"] for name in names])
        qenergy = np.asarray([
            steps[name].attrs.get("quantum_energy", np.nan) for name in names
        ])
        labels = [[_text(value) for value in steps[name]["labels"][:]] for name in names]
        states = [steps[name]["states"][:].astype(int).tolist() for name in names]
        events = json.loads(steps[names[-1]].attrs.get("events_json", "[]"))
        width = int(max(count))
        amplitude_magnitudes = np.full((len(names), width), np.nan)
        tbf_populations = np.full((len(names), width), np.nan)
        for frame, name in enumerate(names):
            group = steps[name]
            coefficients = group["amplitudes"][:]
            overlap = group["S"][:]
            amplitude_magnitudes[frame, :len(coefficients)] = np.abs(coefficients)
            tbf_populations[frame, :len(coefficients)] = np.real(
                np.conjugate(coefficients) * (overlap @ coefficients)
            )
        return {
            "time": time,
            "count": count,
            "populations": populations,
            "amplitude_magnitudes": amplitude_magnitudes,
            "tbf_populations": tbf_populations,
            "norm": norm,
            "quantum_energy": qenergy,
            "labels": labels,
            "states": states,
            "step_names": names,
            "events": events,
        }


def _pyspawn_embedding_residual() -> dict[str, float]:
    residuals: dict[str, float] = {}
    with h5py.File(PYSPAWN_PRISTINE) as left, h5py.File(PYSPAWN) as right:
        for name in ("quantum_time", "num_traj_qm", "qm_amplitudes", "H", "S", "Sdot"):
            residuals[name] = float(np.max(np.abs(
                left[f"sim/{name}"][:] - right[f"sim/{name}"][:]
            )))
        for group in sorted(name for name in left if name.startswith("traj_")):
            for name in ("time", "energies", "positions", "momenta", "timederivcoups"):
                lhs = left[f"{group}/{name}"][:]
                rhs = right[f"{group}/{name}"][:, :lhs.shape[1]]
                residuals[f"{group}/{name}"] = float(np.max(np.abs(lhs - rhs)))
    return residuals


def _common_frames(py: dict, swb: dict):
    py_index = {round(float(value), 8): index for index, value in enumerate(py["time"])}
    for right, value in enumerate(swb["time"]):
        left = py_index.get(round(float(value), 8))
        if left is not None:
            yield left, right


def _trajectory_residuals(swb: dict) -> dict:
    final_swb_labels = swb["labels"][-1]
    with h5py.File(PYSPAWN) as py_handle, h5py.File(SPAWN_WITH_BERRY) as sw_handle:
        py_labels = ["00"] + sorted(
            (name.removeprefix("traj_") for name in py_handle if name.startswith("traj_")),
            key=lambda label: float(py_handle[f"traj_{label}/time"][0, 0]),
        )[1:]
        position_errors = []
        momentum_errors = []
        per_trajectory = []
        for ordinal, (py_label, sw_label) in enumerate(zip(py_labels, final_swb_labels)):
            py_group = py_handle[f"traj_{py_label}"]
            py_times = {
                round(float(value), 8): index
                for index, value in enumerate(py_group["time"][:, 0])
            }
            local_position = []
            local_momentum = []
            for step_name in swb["step_names"]:
                group = sw_handle[f"steps/{step_name}"]
                labels = [_text(value) for value in group["labels"][:]]
                if sw_label not in labels:
                    continue
                py_index = py_times.get(round(float(group.attrs["time"]), 8))
                if py_index is None:
                    continue
                sw_index = labels.index(sw_label)
                local_position.append(float(np.max(np.abs(
                    group["positions"][sw_index, 0, :2]
                    - py_group["positions"][py_index, :2]
                ))))
                local_momentum.append(float(np.max(np.abs(
                    group["momenta"][sw_index, 0, :2]
                    - py_group["momenta"][py_index, :2]
                ))))
            position_errors.extend(local_position)
            momentum_errors.extend(local_momentum)
            per_trajectory.append({
                "ordinal": ordinal,
                "pyspawn_label": py_label,
                "spawnwithberry_label": sw_label,
                "max_position_error": max(local_position, default=np.nan),
                "max_momentum_error": max(local_momentum, default=np.nan),
            })
    return {
        "max_position_error": max(position_errors, default=np.nan),
        "max_momentum_error": max(momentum_errors, default=np.nan),
        "mapping": per_trajectory,
    }


def _matrix_residuals(swb: dict) -> dict:
    overlap_errors = []
    effective_errors = []
    effective_relative_errors = []
    with h5py.File(PYSPAWN) as py_handle, h5py.File(SPAWN_WITH_BERRY) as sw_handle:
        py_times = {
            round(float(value), 8): index
            for index, value in enumerate(py_handle["sim/quantum_time"][:, 0])
        }
        for step_name in swb["step_names"]:
            group = sw_handle[f"steps/{step_name}"]
            py_index = py_times.get(round(float(group.attrs["time"]), 8))
            if py_index is None:
                continue
            ntraj = len(group["states"])
            if int(py_handle["sim/num_traj_qm"][py_index, 0]) != ntraj:
                continue
            py_overlap = py_handle["sim/S"][
                py_index, :ntraj * ntraj
            ].reshape(ntraj, ntraj)
            # PySpawn writes a zero matrix on the instantaneous basis-growth
            # frame before all new centroids exist.  Compare committed full
            # matrices only.
            if np.min(np.real(np.diag(py_overlap))) < 0.5:
                continue
            py_hamiltonian = py_handle["sim/H"][
                py_index, :ntraj * ntraj
            ].reshape(ntraj, ntraj) - ENERGY_SHIFT * py_overlap
            py_tau = py_handle["sim/Sdot"][
                py_index, :ntraj * ntraj
            ].reshape(ntraj, ntraj)
            sw_overlap = group["S"][:]
            py_effective = py_hamiltonian - 1j * py_tau
            sw_effective = group["H"][:] - 1j * group["Sdot"][:]
            # Each newly inserted TBF carries an arbitrary constant phase.  Fix
            # those phases recursively against the strongest already-aligned S
            # or effective-generator edge before comparing matrices.
            phases = np.ones(ntraj, dtype=np.complex128)
            for column in range(1, ntraj):
                candidates = []
                for row in range(column):
                    for py_matrix, sw_matrix in (
                        (py_overlap, sw_overlap),
                        (py_effective, sw_effective),
                    ):
                        if (
                            abs(py_matrix[row, column]) < 1.0e-10
                            or abs(sw_matrix[row, column]) < 1.0e-10
                        ):
                            continue
                        ratio = py_matrix[row, column] / sw_matrix[row, column]
                        candidates.append((
                            abs(py_matrix[row, column]) * abs(sw_matrix[row, column]),
                            phases[row] * ratio / abs(ratio),
                        ))
                if candidates:
                    phases[column] = max(candidates, key=lambda item: item[0])[1]
            gauge = phases.conj()[:, None] * phases[None, :]
            sw_overlap = gauge * sw_overlap
            sw_effective = gauge * sw_effective
            overlap_errors.append(float(np.max(np.abs(py_overlap - sw_overlap))))
            effective_errors.append(float(np.max(np.abs(py_effective - sw_effective))))
            effective_relative_errors.append(float(
                np.linalg.norm(py_effective - sw_effective)
                / max(np.linalg.norm(py_effective), 1.0e-15)
            ))
    return {
        "frames": len(overlap_errors),
        "overlap_max_abs_error": max(overlap_errors, default=np.nan),
        "overlap_median_abs_error": float(np.median(overlap_errors)),
        "effective_generator_max_abs_error": max(effective_errors, default=np.nan),
        "effective_generator_median_abs_error": float(np.median(effective_errors)),
        "effective_generator_max_relative_error": max(
            effective_relative_errors, default=np.nan
        ),
        "effective_generator_median_relative_error": float(
            np.median(effective_relative_errors)
        ),
    }


def _spawn_records(swb: dict) -> tuple[list[dict], list[dict]]:
    swb_spawns = [
        {
            "entry_time": float(event["threshold_entry_time"]),
            "maximum_time": float(event["time"]),
            "coupling": float(event["coupling"]),
            "parent": event["parent"],
            "child": event["child"],
            "target_state": int(event["target_state"]),
        }
        for event in swb["events"] if event.get("kind") == "spawn"
    ]
    py_spawns = []
    threshold = np.pi / 4.0
    with h5py.File(PYSPAWN) as handle:
        child_labels = sorted(
            (name.removeprefix("traj_") for name in handle if name.startswith("traj_")
             and name != "traj_00"),
            key=lambda label: float(handle[f"traj_{label}/time"][0, 0]),
        )
        for child in child_labels:
            parent = child.rsplit("b", 1)[0]
            target = 1 if int(handle[f"traj_{child}/energies"].shape[1]) and child.count("b") == 2 else 0
            # The actual state is unambiguous from the final simulation label map.
            final_labels = swb["labels"][-1]
            del final_labels  # state is read from PySpawn's saved JSON-independent map below
            entry = float(handle[f"traj_{child}/time"][0, 0])
            parent_state = 1 if parent == "00" else 0
            target = 1 - parent_state
            times = handle[f"traj_{parent}/time_half_step"][:, 0]
            coupling = np.abs(handle[f"traj_{parent}/timederivcoups"][:, target])
            window = np.flatnonzero((times >= entry - 0.1) & (times <= entry + 3.0))
            above = window[coupling[window] >= threshold]
            maximum = int(above[np.argmax(coupling[above])])
            py_spawns.append({
                "entry_time": entry,
                "maximum_time": float(times[maximum] + 0.05),
                "coupling": float(coupling[maximum]),
                "parent": parent,
                "child": child,
                "target_state": target,
            })
    return py_spawns, swb_spawns


def main() -> None:
    py = _pyspawn_series(PYSPAWN)
    swb = _swb_series(SPAWN_WITH_BERRY)
    common = list(_common_frames(py, swb))
    py_pop = np.asarray([py["populations"][left] for left, _ in common])
    swb_pop = np.asarray([swb["populations"][right] for _, right in common])
    population_error = swb_pop - py_pop
    amplitude_errors = []
    tbf_population_errors = []
    for left, right in common:
        count = min(int(py["count"][left]), int(swb["count"][right]))
        amplitude_errors.extend(np.abs(
            swb["amplitude_magnitudes"][right, :count]
            - py["amplitude_magnitudes"][left, :count]
        ))
        tbf_population_errors.extend(np.abs(
            swb["tbf_populations"][right, :count]
            - py["tbf_populations"][left, :count]
        ))
    py_spawns, swb_spawns = _spawn_records(swb)
    spawn_count = min(len(py_spawns), len(swb_spawns))
    report = {
        "reference": "PySpawn native fulldiag",
        "candidate": "SpawnWithBerry adaptive metric Cayley",
        "common_frames": len(common),
        "pyspawn_final_tbf_count": int(py["count"][-1]),
        "spawnwithberry_final_tbf_count": int(swb["count"][-1]),
        "spawn_count_match": len(py_spawns) == len(swb_spawns),
        "maximum_spawn_entry_time_error_au": max(
            (abs(py_spawns[i]["entry_time"] - swb_spawns[i]["entry_time"])
             for i in range(spawn_count)), default=np.nan,
        ),
        "maximum_spawn_peak_time_error_au": max(
            (abs(py_spawns[i]["maximum_time"] - swb_spawns[i]["maximum_time"])
             for i in range(spawn_count)), default=np.nan,
        ),
        "maximum_spawn_coupling_error": max(
            (abs(py_spawns[i]["coupling"] - swb_spawns[i]["coupling"])
             for i in range(spawn_count)), default=np.nan,
        ),
        "state_population_max_abs_error": np.max(np.abs(population_error), axis=0).tolist(),
        "state_population_rms_error": np.sqrt(np.mean(population_error**2, axis=0)).tolist(),
        "tbf_amplitude_magnitude_max_abs_error": float(max(amplitude_errors)),
        "tbf_amplitude_magnitude_rms_error": float(np.sqrt(np.mean(
            np.square(amplitude_errors)
        ))),
        "tbf_population_max_abs_error": float(max(tbf_population_errors)),
        "tbf_population_rms_error": float(np.sqrt(np.mean(
            np.square(tbf_population_errors)
        ))),
        "final_populations": {
            "pyspawn": py["populations"][-1].tolist(),
            "spawnwithberry": swb["populations"][-1].tolist(),
        },
        "metric_norm": {
            "pyspawn_max_abs_error": float(np.max(np.abs(py["norm"] - 1.0))),
            "spawnwithberry_max_abs_error": float(np.max(np.abs(swb["norm"] - 1.0))),
            "pyspawn_final": float(py["norm"][-1]),
            "spawnwithberry_final": float(swb["norm"][-1]),
        },
        "quantum_energy_max_drift": {
            "pyspawn": float(np.nanmax(np.abs(
                py["quantum_energy"] - py["quantum_energy"][0]
            ))),
            "spawnwithberry": float(np.nanmax(np.abs(
                swb["quantum_energy"] - swb["quantum_energy"][0]
            ))),
        },
        "quantum_energy_final_drift": {
            "pyspawn": float(py["quantum_energy"][-1] - py["quantum_energy"][0]),
            "spawnwithberry": float(
                swb["quantum_energy"][-1] - swb["quantum_energy"][0]
            ),
        },
        "matrix_residuals": _matrix_residuals(swb),
        "trajectory_residuals": _trajectory_residuals(swb),
        "pyspawn_2d_vs_inert_3d_max_residual": max(
            _pyspawn_embedding_residual().values()
        ),
        "pyspawn_spawns": py_spawns,
        "spawnwithberry_spawns": swb_spawns,
    }
    (ROOT / "comparison-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    with (ROOT / "comparison-series.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("time_au", "pyspawn_s0", "pyspawn_s1", "spawnwithberry_s0",
                         "spawnwithberry_s1", "pyspawn_norm", "spawnwithberry_norm",
                         "pyspawn_tbfs", "spawnwithberry_tbfs"))
        for left, right in common:
            writer.writerow((
                swb["time"][right], *py["populations"][left], *swb["populations"][right],
                py["norm"][left], swb["norm"][right], py["count"][left], swb["count"][right],
            ))
    try:
        import matplotlib.pyplot as plt

        time = np.asarray([swb["time"][right] for _, right in common])
        py_norm = np.asarray([py["norm"][left] for left, _ in common])
        swb_norm = np.asarray([swb["norm"][right] for _, right in common])
        py_count = np.asarray([py["count"][left] for left, _ in common])
        swb_count = np.asarray([swb["count"][right] for _, right in common])
        figure, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        axes[0].plot(time, py_pop[:, 0], label="PySpawn S0", linewidth=1.5)
        axes[0].plot(time, swb_pop[:, 0], "--", label="SpawnWithBerry S0", linewidth=1.5)
        axes[0].plot(time, py_pop[:, 1], label="PySpawn S1", linewidth=1.0, alpha=0.7)
        axes[0].plot(time, swb_pop[:, 1], "--", label="SpawnWithBerry S1", linewidth=1.0, alpha=0.7)
        axes[0].set_ylabel("state population")
        axes[0].legend(ncol=2, fontsize=8)
        axes[1].plot(time, swb_pop[:, 0] - py_pop[:, 0])
        axes[1].axhline(0.0, color="0.5", linewidth=0.7)
        axes[1].set_ylabel("S0 difference")
        axes[2].semilogy(time, np.maximum(np.abs(py_norm - 1.0), 1.0e-16), label="PySpawn")
        axes[2].semilogy(time, np.maximum(np.abs(swb_norm - 1.0), 1.0e-16), label="SpawnWithBerry")
        axes[2].set_ylabel("|S-norm - 1|")
        axes[2].legend(fontsize=8)
        axes[3].step(time, py_count, where="post", label="PySpawn")
        axes[3].step(time, swb_count, "--", where="post", label="SpawnWithBerry")
        axes[3].set_ylabel("TBF count")
        axes[3].set_xlabel("time / au")
        axes[3].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(ROOT / "comparison.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
