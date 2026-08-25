"""Incremental, replay-safe human-readable mirrors of scientific history."""

from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from ..config import AU_TIME_PER_FS, BOHR_PER_ANGSTROM


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _safe_filename(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    ).strip(".")
    return safe or "tbf"


def _value(value: Any, default: float = float("nan")) -> Any:
    if value is None:
        return default
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _append_csv(path: Path, headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if not exists:
            writer.writerow(headers)
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(type(value).__name__)


def _committed_names(handle: h5py.File) -> list[str]:
    steps = handle["steps"]
    committed = int(steps.attrs.get("committed_through", -1))
    return sorted((name for name in steps if int(name) <= committed), key=int)


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _tbf_directory(root: Path, label: str, identifier: str) -> Path:
    return root / "tbfs" / f"{_safe_filename(label)}-{_safe_filename(identifier[:12])}"


def _write_readme(root: Path) -> None:
    _atomic_text(
        root / "README.txt",
        """AIMS-Berry live readable output

simulation.h5 remains the authoritative, full-precision scientific history.
These CSV/JSON files mirror committed HDF5 frames only; replay-staged frames are
excluded. Times and quantities use atomic units unless a column says otherwise.

populations.csv
  Coherent electronic-state populations. Their sum equals the metric norm.
quantum_diagnostics.csv
  Norm, quantum energy, drift, integration, gap, coupling, and basis diagnostics.
tbfs/index.csv
  Stable TBF identity, label, parent, birth time, and initial electronic state.
tbfs/<label-id>/energies.csv
  Absolute electronic energies and per-TBF classical KE/PE/total/reference/drift.
tbfs/<label-id>/phase_space.csv
  Complex coefficient and every nuclear position/momentum component.
tbfs/<label-id>/couplings.csv
  Complex projected derivative couplings d_IJ dot v and their magnitudes.
tbfs/<label-id>/derivative_norms.csv
  Norms of each evaluated gradient and NAC vector; full arrays remain in HDF5.
events.jsonl and spawns.csv
  Task/spawn/replay history and a compact spawn-only table.

coefficient_abs2 is |c_N|^2, not a physical standalone TBF population in a
nonorthogonal Gaussian basis. gross_tbf_population is Re[c_N* (S c)_N], the
partition used by spawning decisions; it sums to the metric norm but individual
terms need not behave like probabilities. Use populations.csv for coherent state
populations.
""",
    )


def _append_frame(root: Path, group: h5py.Group, atoms: tuple[str, ...]) -> None:
    step = int(group.name.rsplit("/", 1)[-1])
    time_au = float(group.attrs["time"])
    time_fs = time_au / AU_TIME_PER_FS
    labels = [_text(value) for value in group["labels"][:]]
    identifiers = [_text(value) for value in group["ids"][:]]
    parents = (
        [_text(value) for value in group["parents"][:]]
        if "parents" in group
        else [""] * len(labels)
    )
    states = group["states"][:].astype(int)
    amplitudes = group["amplitudes"][:]
    gross_tbf_populations = (
        np.real(np.conjugate(amplitudes) * (group["S"][:] @ amplitudes))
        if "S" in group
        else np.abs(amplitudes) ** 2
    )
    positions = group["positions"][:]
    momenta = group["momenta"][:]
    energies = group["energies"][:]
    populations = group["populations"][:]

    _append_csv(
        root / "populations.csv",
        ("step", "time_au", "time_fs", *(f"state_{i}" for i in range(len(populations)))),
        ((step, time_au, time_fs, *populations.tolist()),),
    )

    couplings = group["projected_couplings"][:] if "projected_couplings" in group else None
    finite_couplings = (
        np.abs(couplings[np.isfinite(couplings)]) if couplings is not None else np.asarray([])
    )
    if energies.shape[1] > 1:
        upper = np.triu_indices(energies.shape[1], k=1)
        gaps = np.abs(energies[:, upper[0]] - energies[:, upper[1]]).ravel()
        finite_gaps = gaps[np.isfinite(gaps)]
    else:
        finite_gaps = np.asarray([])
    retained = group["retained_overlap_eigenvalues"][:] if "retained_overlap_eigenvalues" in group else np.asarray([])
    population_sum = float(np.sum(populations))
    quantum_headers = (
        "step", "time_au", "time_fs", "metric_norm", "state_population_sum",
        "quantum_energy_hartree", "quantum_reference_hartree",
        "quantum_drift_hartree", "raw_norm_before", "raw_norm_after",
        "quantum_substeps", "coefficient_error", "metric_residual",
        "minimum_gap_hartree", "maximum_projected_coupling_au", "tbf_count",
        "minimum_retained_overlap_eigenvalue", "hermiticity_residual",
        "nuclear_time_step_au",
    )
    quantum_row = (
        step, time_au, time_fs, _value(group.attrs.get("metric_norm")), population_sum,
        _value(group.attrs.get("quantum_energy")),
        _value(group.attrs.get("quantum_energy_reference")),
        _value(group.attrs.get("quantum_energy_drift")),
        _value(group.attrs.get("raw_norm_before")),
        _value(group.attrs.get("raw_norm_after")),
        _value(group.attrs.get("quantum_substeps")),
        _value(group.attrs.get("quantum_convergence_error")),
        _value(group.attrs.get("metric_compatibility_residual")),
        float(np.min(finite_gaps)) if finite_gaps.size else float("nan"),
        float(np.max(finite_couplings)) if finite_couplings.size else float("nan"),
        len(labels), float(np.min(retained)) if retained.size else float("nan"),
        _value(group.attrs.get("hamiltonian_hermiticity_residual")),
        _value(group.attrs.get("nuclear_time_step")),
    )
    _append_csv(root / "quantum_diagnostics.csv", quantum_headers, (quantum_row,))

    kinetic = group["classical_kinetic_energy"][:]
    potential = group["classical_potential_energy"][:]
    total = group["classical_total_energy"][:]
    references = group["classical_reference_energy"][:]
    drifts = group["classical_energy_drift"][:]
    gradient_mask = group["gradient_mask"][:] if "gradient_mask" in group else None
    nac_mask = group["nac_mask"][:] if "nac_mask" in group else None
    gradients = group["gradients"] if "gradients" in group else None
    nacs = group["nacs"] if "nacs" in group else None

    for index, (label, identifier, parent, state) in enumerate(
        zip(labels, identifiers, parents, states, strict=True)
    ):
        directory = _tbf_directory(root, label, identifier)
        index_path = root / "tbfs" / "index.csv"
        if not directory.exists():
            _append_csv(
                index_path,
                ("identifier", "label", "parent_identifier", "birth_step", "birth_time_au", "initial_state", "directory"),
                ((identifier, label, parent, step, time_au, int(state), directory.name),),
            )
        energy_headers = (
            "step", "time_au", "time_fs", "active_state",
            "kinetic_hartree", "active_potential_hartree", "classical_total_hartree",
            "classical_reference_hartree", "classical_drift_hartree",
            *(f"electronic_state_{number}_hartree" for number in range(energies.shape[1])),
        )
        _append_csv(
            directory / "energies.csv",
            energy_headers,
            ((
                step, time_au, time_fs, int(state), kinetic[index], potential[index],
                total[index], references[index], drifts[index], *energies[index].tolist(),
            ),),
        )

        coordinate_headers: list[str] = []
        coordinate_values: list[float] = []
        for atom_index, atom in enumerate(atoms):
            for axis, component in zip("xyz", positions[index, atom_index], strict=True):
                coordinate_headers.append(f"{atom}{atom_index}_{axis}_angstrom")
                coordinate_values.append(float(component / BOHR_PER_ANGSTROM))
            for axis, component in zip("xyz", momenta[index, atom_index], strict=True):
                coordinate_headers.append(f"{atom}{atom_index}_p{axis}_au")
                coordinate_values.append(float(component))
        _append_csv(
            directory / "phase_space.csv",
            (
                "step", "time_au", "time_fs", "active_state", "amplitude_real",
                "amplitude_imag", "coefficient_abs2", "gross_tbf_population",
                *coordinate_headers,
            ),
            ((
                step, time_au, time_fs, int(state), amplitudes[index].real,
                amplitudes[index].imag, abs(amplitudes[index]) ** 2,
                gross_tbf_populations[index], *coordinate_values,
            ),),
        )

        if couplings is not None:
            coupling_rows = []
            for target, coupling in enumerate(couplings[index]):
                if np.isfinite(coupling):
                    coupling_rows.append((
                        step, time_au, time_fs, int(state), target,
                        coupling.real, coupling.imag, abs(coupling),
                    ))
            _append_csv(
                directory / "couplings.csv",
                (
                    "step", "time_au", "time_fs", "active_state", "target_state",
                    "projected_coupling_real_au", "projected_coupling_imag_au",
                    "projected_coupling_abs_au",
                ),
                coupling_rows,
            )

        derivative_rows = []
        if gradients is not None and gradient_mask is not None:
            for gradient_state in np.flatnonzero(gradient_mask[index]):
                derivative_rows.append((
                    step, time_au, time_fs, "gradient", int(gradient_state), "",
                    float(np.linalg.norm(gradients[index, gradient_state])),
                ))
        if nacs is not None and nac_mask is not None:
            for left, right in np.argwhere(nac_mask[index]):
                derivative_rows.append((
                    step, time_au, time_fs, "nac", int(left), int(right),
                    float(np.linalg.norm(nacs[index, left, right])),
                ))
        _append_csv(
            directory / "derivative_norms.csv",
            ("step", "time_au", "time_fs", "kind", "state_i", "state_j", "norm_au"),
            derivative_rows,
        )


def _write_events(root: Path, events: list[dict[str, Any]]) -> None:
    event_text = "".join(
        json.dumps(event, default=_json_default, sort_keys=True) + "\n" for event in events
    )
    _atomic_text(root / "events.jsonl", event_text)
    spawn_rows = []
    for event in events:
        if event.get("kind") not in {"spawn", "failed_spawn"}:
            continue
        spawn_rows.append((
            event.get("time", ""), event.get("kind", ""), event.get("parent", ""),
            event.get("child", ""), event.get("target_state", ""),
            event.get("coupling", ""), event.get("reason", ""),
        ))
    target = root / "spawns.csv"
    temporary = target.with_name(f".{target.name}-{uuid.uuid4().hex}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_au", "kind", "parent", "child", "target_state", "coupling", "reason"))
        writer.writerows(spawn_rows)
    os.replace(temporary, target)


def _write_status(root: Path, handle: h5py.File, names: list[str]) -> None:
    version = {
        key: _value(handle.attrs[key])
        for key in (
            "aims_berry_version", "aims_berry_last_update", "aims_berry_commit",
            "aims_berry_dirty", "aims_berry_source",
        )
        if key in handle.attrs
    }
    payload: dict[str, Any] = {
        "history": str(Path(handle.filename).resolve()),
        "committed_frames": len(names),
        "last_step": int(names[-1]) if names else None,
        "last_time_au": float(handle[f"steps/{names[-1]}"].attrs["time"]) if names else None,
        "version": version,
    }
    if names:
        group = handle[f"steps/{names[-1]}"]
        payload.update({
            "tbf_count": len(group["states"]),
            "states": group["states"][:].astype(int).tolist(),
            "populations": group["populations"][:].tolist(),
            "metric_norm": _value(group.attrs.get("metric_norm")),
            "quantum_energy_hartree": _value(group.attrs.get("quantum_energy")),
            "quantum_energy_drift_hartree": _value(group.attrs.get("quantum_energy_drift")),
        })
    _atomic_text(root / "run_status.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _populate(root: Path, handle: h5py.File, names: list[str], atoms: tuple[str, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_readme(root)
    for name in names:
        _append_frame(root, handle[f"steps/{name}"], atoms)
    events: list[dict[str, Any]] = []
    if names:
        events = json.loads(handle[f"steps/{names[-1]}"].attrs.get("events_json", "[]"))
    _write_events(root, events)
    _write_status(root, handle, names)
    _atomic_text(
        root / "manifest.json",
        json.dumps({"schema_version": 1, "exported_steps": names}, indent=2) + "\n",
    )


def _rebuild(root: Path, handle: h5py.File, names: list[str], atoms: tuple[str, ...]) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.with_name(f".{root.name}-{uuid.uuid4().hex}.tmp")
    backup = root.with_name(f".{root.name}-{uuid.uuid4().hex}.old")
    _populate(temporary, handle, names, atoms)
    if root.exists():
        os.replace(root, backup)
    os.replace(temporary, root)
    if backup.exists():
        shutil.rmtree(backup)


def export_readable_history(
    history: str | Path,
    output_directory: str | Path,
    *,
    changed_step: int | None = None,
) -> Path:
    """Synchronize a readable mirror with committed HDF5 history.

    Ordinary new frames append in constant work. Rollback, replay commit, a
    same-step overwrite, or an interrupted prior export triggers an atomic rebuild.
    """

    history = Path(history)
    if history.is_dir():
        history = history / "simulation.h5"
    root = Path(output_directory)
    in_progress = root / ".sync-in-progress"
    with h5py.File(history, "r") as handle:
        if "steps" not in handle:
            raise ValueError(f"{history} is not an aims_berry history")
        names = _committed_names(handle)
        atoms = tuple(_text(value) for value in handle["atoms"][:])
        manifest = _read_manifest(root)
        exported = [str(name) for name in manifest.get("exported_steps", [])]
        must_rebuild = (
            in_progress.exists()
            or exported != names[: len(exported)]
            or (changed_step is not None and f"{changed_step:08d}" in exported)
        )
        if must_rebuild:
            _rebuild(root, handle, names, atoms)
            return root
        if exported == names:
            _write_status(root, handle, names)
            return root

        root.mkdir(parents=True, exist_ok=True)
        _atomic_text(in_progress, "Readable export is being synchronized.\n")
        try:
            if not exported:
                _write_readme(root)
            for name in names[len(exported):]:
                _append_frame(root, handle[f"steps/{name}"], atoms)
            events: list[dict[str, Any]] = []
            if names:
                events = json.loads(
                    handle[f"steps/{names[-1]}"].attrs.get("events_json", "[]")
                )
            _write_events(root, events)
            _write_status(root, handle, names)
            _atomic_text(
                root / "manifest.json",
                json.dumps({"schema_version": 1, "exported_steps": names}, indent=2) + "\n",
            )
        finally:
            in_progress.unlink(missing_ok=True)
    return root
