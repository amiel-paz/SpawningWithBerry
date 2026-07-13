"""Read and reproduce observables from the public PySpawn HDF5 schema."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from ..config import AU_TIME_PER_FS, BOHR_PER_ANGSTROM
from .dataset import bond

HARTREE_TO_EV = 27.211386245988
PYSPAWN_PAPER_HARTREE_TO_EV = 13.6


def unsigned_dihedral(
    geometry: np.ndarray, i: int, j: int, k: int, last: int
) -> float:
    """Return the unsigned plane angle used by the original PySpawn analysis."""
    rji = geometry[i] - geometry[j]
    rjk = geometry[k] - geometry[j]
    rkj = -rjk
    rkl = geometry[last] - geometry[k]
    normal_1 = np.cross(rji, rjk)
    normal_2 = np.cross(rkj, rkl)
    cosine = np.dot(normal_1, normal_2) / (
        np.linalg.norm(normal_1) * np.linalg.norm(normal_2)
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class PySpawnDataset:
    """Adapter for the PyTables/HDF5 layout shipped with the PySpawn paper."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with h5py.File(self.path, "r") as handle:
            if "sim/qm_amplitudes" not in handle or "traj_00/positions" not in handle:
                raise ValueError(f"{self.path} is not a recognized PySpawn history")
            self.labels = tuple(self._decode(value) for value in handle["sim"].attrs["labels"])
            self.states = tuple(int(value) for value in handle["sim"].attrs["istates"])
            self.atoms = tuple(self._decode(value) for value in handle["traj_00"].attrs["atoms"])

    @staticmethod
    def _decode(value) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def trajectory(self, label: str) -> dict[str, np.ndarray]:
        with h5py.File(self.path, "r") as handle:
            group = handle[f"traj_{label}"]
            geometry = group["positions"][...].reshape(-1, len(self.atoms), 3)
            return {
                "time_fs": group["time"][:, 0] / AU_TIME_PER_FS,
                "geometry_angstrom": geometry / BOHR_PER_ANGSTROM,
                "energies_ev": group["energies"][...] * HARTREE_TO_EV,
            }

    def coordinate(self, kind: str, atoms: tuple[int, ...]) -> dict[str, np.ndarray]:
        function = {"bond": bond, "dihedral": unsigned_dihedral}[kind]
        output = {}
        for label in self.labels:
            trajectory = self.trajectory(label)
            values = [function(geometry, *atoms) for geometry in trajectory["geometry_angstrom"]]
            output[label] = np.column_stack((trajectory["time_fs"], values))
        return output

    def energy_gap(
        self,
        label: str = "00",
        lower: int = 0,
        upper: int = 1,
        *,
        convention: Literal["physical", "pyspawn-paper"] = "physical",
    ) -> np.ndarray:
        """Return a state gap, preserving the paper's legacy conversion on request.

        PySpawn's published plotting helper used ``13.6`` to label Hartree energy
        differences as eV.  The physically correct conversion is the default;
        ``pyspawn-paper`` exists solely to reproduce Figure 7's vertical scale.
        """
        trajectory = self.trajectory(label)
        conversion = (
            HARTREE_TO_EV
            if convention == "physical"
            else PYSPAWN_PAPER_HARTREE_TO_EV
        )
        gap = (
            trajectory["energies_ev"][:, upper]
            - trajectory["energies_ev"][:, lower]
        ) * (conversion / HARTREE_TO_EV)
        return np.column_stack((trajectory["time_fs"], gap))

    def populations(self) -> np.ndarray:
        """Coherent state populations ``c_I^* S_II c_I`` in published label order."""
        with h5py.File(self.path, "r") as handle:
            times = handle["sim/quantum_time"][:, 0] / AU_TIME_PER_FS
            counts = handle["sim/num_traj_qm"][:, 0].astype(int)
            amplitudes = handle["sim/qm_amplitudes"]
            overlaps = handle["sim/S"]
            nstates = max(self.states) + 1
            populations = np.zeros((len(times), nstates), dtype=float)
            for row, count in enumerate(counts):
                coefficients = amplitudes[row, :count]
                overlap = overlaps[row, : count * count].reshape(count, count)
                active_states = self.states[:count]
                for state in range(nstates):
                    indices = [index for index, value in enumerate(active_states) if value == state]
                    if indices:
                        block = overlap[np.ix_(indices, indices)]
                        vector = coefficients[indices]
                        populations[row, state] = float(np.real(np.vdot(vector, block @ vector)))
        return np.column_stack((times, populations))

    def summary(self) -> dict[str, Any]:
        bonds = self.coordinate("bond", (3, 11))
        dihedrals = self.coordinate("dihedral", (2, 6, 9, 10))
        gap = self.energy_gap()
        paper_gap = self.energy_gap(convention="pyspawn-paper")
        populations = self.populations()
        spawn_times = {label: float(values[0, 0]) for label, values in bonds.items() if label != "00"}
        minimum = int(np.argmin(gap[:, 1]))
        return {
            "source_schema": "pyspawn-paper-sim.hdf5",
            "trajectory_count": len(self.labels),
            "quantum_frames": len(populations),
            "initial_bond_angstrom": float(bonds["00"][0, 1]),
            "initial_dihedral_degree": float(dihedrals["00"][0, 1]),
            "minimum_gap_ev": float(gap[minimum, 1]),
            "minimum_gap_time_fs": float(gap[minimum, 0]),
            "pyspawn_paper_minimum_gap_labeled_ev": float(paper_gap[minimum, 1]),
            "pyspawn_paper_energy_conversion_hartree_to_ev": (
                PYSPAWN_PAPER_HARTREE_TO_EV
            ),
            "final_parent_bond_angstrom": float(bonds["00"][-1, 1]),
            "final_parent_dihedral_degree": float(dihedrals["00"][-1, 1]),
            "final_representative_s1_population": float(populations[-1, 2]),
            "population_norm_max_error": float(np.max(np.abs(populations[:, 1:].sum(axis=1) - 1.0))),
            "spawn_times_fs": spawn_times,
        }


def _write_series(path: Path, headers: tuple[str, ...], values: np.ndarray) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(values.tolist())


def analyze_pyspawn_reference(path: str | Path, output_directory: str | Path) -> list[Path]:
    """Generate Figure 6c/6d/7-style datasets and the representative population trace."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires the 'analysis' optional dependency") from exc

    dataset = PySpawnDataset(path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    products: list[Path] = []
    specifications = (
        ("figure6c_distance", dataset.coordinate("bond", (3, 11)), "Distance (Å)"),
        ("figure6d_dihedral", dataset.coordinate("dihedral", (2, 6, 9, 10)), "Angle (degree)"),
    )
    for name, series, ylabel in specifications:
        figure, axis = plt.subplots(figsize=(5.2, 3.8))
        for label, values in series.items():
            axis.plot(values[:, 0], values[:, 1], label=label)
            _write_series(output / f"{name}_{label}.csv", ("time_fs", ylabel), values)
        axis.set(xlabel="Time (fs)", ylabel=ylabel)
        axis.legend()
        target = output / f"{name}.png"
        figure.savefig(target, dpi=180, bbox_inches="tight")
        plt.close(figure)
        products.append(target)

    physical_gap = dataset.energy_gap()
    _write_series(
        output / "figure7_gap_physical.csv", ("time_fs", "gap_ev"), physical_gap
    )
    gap = dataset.energy_gap(convention="pyspawn-paper")
    _write_series(
        output / "figure7_gap.csv", ("time_fs", "paper_labeled_gap_ev"), gap
    )
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    axis.plot(gap[:, 0], gap[:, 1], color="tab:green", label="00: S1-S0")
    axis.set(xlabel="Time (fs)", ylabel="Energy gap (paper eV convention)")
    axis.legend()
    target = output / "figure7_gap.png"
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    products.append(target)

    populations = dataset.populations()
    _write_series(output / "representative_populations.csv", ("time_fs", "S0", "S1"), populations)
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    axis.plot(populations[:, 0], populations[:, 2], color="black", label="representative S1")
    axis.set(xlabel="Time (fs)", ylabel="Population on S1", ylim=(-0.02, 1.02))
    axis.legend()
    target = output / "representative_population.png"
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)
    products.append(target)

    (output / "reference_summary.json").write_text(
        json.dumps(dataset.summary(), indent=2, sort_keys=True) + "\n"
    )
    return products
