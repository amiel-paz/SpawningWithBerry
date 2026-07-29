"""HDF5 history and atomic restart checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .._version import __version__
from ..config import BOHR_PER_ANGSTROM, SimulationConfig
from ..core import SimulationState
from ..electronic.base import ElectronicStructureProvider
from ..geometry import read_xyz
from ..tasks import TaskQueue


class HDF5Writer:
    def __init__(
        self,
        path: str | Path,
        config: SimulationConfig,
        atoms: tuple[str, ...] | None = None,
        *,
        restart: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.atoms = tuple(atoms or read_xyz(config.geometry, config.geometry_units)[0])
        self.xyz_root = self.path.parent / "geometries" if config.write_xyz else None
        mode = "a" if restart else "w"
        self.replay_window: str | None = None
        with h5py.File(self.path, mode, libver="latest") as handle:
            handle.attrs["schema_version"] = 3
            handle.attrs["aims_berry_version"] = __version__
            handle.attrs["units"] = "atomic"
            handle.attrs["config_json"] = json.dumps(config.to_dict(), sort_keys=True)
            strings = h5py.string_dtype("utf-8")
            if "atoms" in handle:
                stored_atoms = tuple(_text(value) for value in handle["atoms"][:])
                if stored_atoms != self.atoms:
                    raise ValueError(
                        f"history atoms {stored_atoms} do not match input atoms {self.atoms}"
                    )
            else:
                handle.create_dataset(
                    "atoms", data=np.asarray(self.atoms, dtype=strings)
                )
            steps = handle.require_group("steps")
            handle.require_group("replay")
            if "committed_through" not in steps.attrs:
                steps.attrs["committed_through"] = max(
                    (int(name) for name in steps), default=-1
                )
        if self.xyz_root is not None:
            export_xyz_history(self.path, self.xyz_root)

    def begin_replay(
        self, window: str, *, entry_step: int, frontier_step: int
    ) -> None:
        """Start or resume an isolated replay-history transaction."""

        self.replay_window = str(window)
        with h5py.File(self.path, "a", libver="latest") as handle:
            root = handle["replay"].require_group(self.replay_window)
            root.attrs["entry_step"] = int(entry_step)
            root.attrs["frontier_step"] = int(frontier_step)
            root.require_group("steps")
            superseded = root.require_group("superseded_steps")
            canonical = handle["steps"]
            for name in sorted(list(canonical)):
                if int(name) <= int(entry_step):
                    continue
                if name in superseded:
                    del superseded[name]
                handle.move(f"steps/{name}", f"replay/{self.replay_window}/superseded_steps/{name}")
            handle["steps"].attrs["committed_through"] = int(entry_step)
            handle.attrs["active_replay"] = self.replay_window
            handle.flush()
        if self.xyz_root is not None:
            export_xyz_history(self.path, self.xyz_root)

    def reset_replay(self, window: str) -> None:
        """Discard provisional frames but retain the replay transaction metadata."""

        self.replay_window = str(window)
        with h5py.File(self.path, "a", libver="latest") as handle:
            root = handle["replay"].require_group(self.replay_window)
            if "steps" in root:
                del root["steps"]
            root.create_group("steps")
            handle.attrs["active_replay"] = self.replay_window
            handle.flush()

    def commit_replay(self, window: str, frontier_step: int) -> None:
        """Atomically expose a completed replay window to analysis readers."""

        window = str(window)
        with h5py.File(self.path, "a", libver="latest") as handle:
            staged = handle[f"replay/{window}/steps"]
            canonical = handle["steps"]
            for name in sorted(staged):
                if name in canonical:
                    del canonical[name]
                handle.move(f"replay/{window}/steps/{name}", f"steps/{name}")
            canonical.attrs["committed_through"] = int(frontier_step)
            del handle[f"replay/{window}"]
            if "active_replay" in handle.attrs:
                del handle.attrs["active_replay"]
            handle.flush()
        self.replay_window = None
        if self.xyz_root is not None:
            export_xyz_history(self.path, self.xyz_root)

    def write_step(
        self,
        state: SimulationState,
        num_states: int,
        diagnostics: dict[str, Any] | None = None,
        classical_energy_references: dict[str, float] | None = None,
        quantum_energy_reference: float | None = None,
    ) -> None:
        with h5py.File(self.path, "a", libver="latest") as handle:
            steps = (
                handle[f"replay/{self.replay_window}/steps"]
                if self.replay_window is not None
                else handle["steps"]
            )
            name = f"{state.step:08d}"
            if name in steps:
                del steps[name]
            group = steps.create_group(name)
            trajectories = state.trajectories
            group.attrs["time"] = state.quantum_time
            group.create_dataset("positions", data=np.asarray([t.positions for t in trajectories]))
            group.create_dataset("momenta", data=np.asarray([t.momenta for t in trajectories]))
            group.create_dataset("widths", data=np.asarray([t.widths for t in trajectories]))
            group.create_dataset("masses", data=np.asarray([t.masses for t in trajectories]))
            group.create_dataset("states", data=np.asarray([t.state for t in trajectories], dtype=int))
            strings = h5py.string_dtype("utf-8")
            group.create_dataset("labels", data=np.asarray([t.label for t in trajectories], dtype=strings))
            group.create_dataset("ids", data=np.asarray([t.identifier for t in trajectories], dtype=strings))
            group.create_dataset("parents", data=np.asarray([t.parent_id or "" for t in trajectories], dtype=strings))
            group.create_dataset("amplitudes", data=state.amplitudes)
            group.create_dataset("populations", data=state.populations(num_states=num_states))
            group.attrs["metric_norm"] = (
                float(np.real(np.vdot(state.amplitudes, state.matrices.overlap @ state.amplitudes)))
                if state.matrices is not None else float(np.vdot(state.amplitudes, state.amplitudes).real)
            )
            energies = np.full((len(trajectories), num_states), np.nan)
            gradients = np.full((len(trajectories), num_states, trajectories[0].positions.shape[0], 3), np.nan)
            nacs = np.full(
                (len(trajectories), num_states, num_states, trajectories[0].positions.shape[0], 3),
                np.nan + 0j,
            )
            gradient_masks = np.zeros((len(trajectories), num_states), dtype=bool)
            nac_masks = np.zeros((len(trajectories), num_states, num_states), dtype=bool)
            state_overlaps = np.full(
                (len(trajectories), num_states, num_states), np.nan + 0j
            )
            for index, trajectory in enumerate(trajectories):
                if trajectory.electronic is not None:
                    energies[index] = trajectory.electronic.energies
                    if trajectory.electronic.gradients is not None:
                        gradients[index] = trajectory.electronic.gradients
                        gradient_masks[index] = trajectory.electronic.gradient_mask
                    if trajectory.electronic.nacs is not None:
                        nacs[index] = trajectory.electronic.nacs
                        nac_masks[index] = trajectory.electronic.nac_mask
                    if trajectory.electronic.state_overlaps is not None:
                        state_overlaps[index] = trajectory.electronic.state_overlaps
            group.create_dataset("energies", data=energies)
            group.create_dataset("gradients", data=gradients, compression="gzip")
            group.create_dataset("nacs", data=nacs, compression="gzip")
            group.create_dataset("gradient_mask", data=gradient_masks)
            group.create_dataset("nac_mask", data=nac_masks)
            group.create_dataset("state_overlaps", data=state_overlaps)
            velocities = np.asarray([t.momenta / t.masses for t in trajectories])
            kinetic = np.asarray([
                np.sum(t.momenta * t.momenta / (2.0 * t.masses)) for t in trajectories
            ])
            potential = np.asarray([
                energies[index, trajectory.state]
                for index, trajectory in enumerate(trajectories)
            ])
            group.create_dataset("classical_kinetic_energy", data=kinetic)
            group.create_dataset("classical_potential_energy", data=potential)
            group.create_dataset("classical_total_energy", data=kinetic + potential)
            references = np.asarray([
                (
                    classical_energy_references.get(
                        trajectory.identifier, kinetic[index] + potential[index]
                    )
                    if classical_energy_references is not None
                    else kinetic[index] + potential[index]
                )
                for index, trajectory in enumerate(trajectories)
            ])
            group.create_dataset("classical_reference_energy", data=references)
            group.create_dataset(
                "classical_energy_drift", data=kinetic + potential - references
            )
            projected = np.full((len(trajectories), num_states), np.nan + 0j)
            for index, trajectory in enumerate(trajectories):
                if trajectory.electronic is None or trajectory.electronic.nacs is None:
                    continue
                for target in range(num_states):
                    if (
                        trajectory.electronic.nac_mask is None
                        or not trajectory.electronic.nac_mask[trajectory.state, target]
                    ):
                        continue
                    projected[index, target] = np.sum(
                        trajectory.electronic.nac_between(trajectory.state, target)
                        * velocities[index]
                    )
            group.create_dataset("projected_couplings", data=projected)
            group.create_dataset("phases", data=np.asarray([t.phase for t in trajectories]))
            group.attrs["electronic_metadata_json"] = json.dumps(
                [t.electronic.metadata if t.electronic is not None else {} for t in trajectories],
                default=_json_default,
            )
            if state.matrices is not None:
                group.create_dataset("H", data=state.matrices.hamiltonian)
                group.create_dataset("S", data=state.matrices.overlap)
                group.create_dataset("Sdot", data=state.matrices.sdot)
                denominator = np.vdot(state.amplitudes, state.matrices.overlap @ state.amplitudes)
                quantum_energy = float(np.real(
                    np.vdot(state.amplitudes, state.matrices.hamiltonian @ state.amplitudes)
                    / denominator
                ))
                reference = (
                    quantum_energy
                    if quantum_energy_reference is None
                    else float(quantum_energy_reference)
                )
                group.attrs["quantum_energy"] = quantum_energy
                group.attrs["quantum_energy_reference"] = reference
                group.attrs["quantum_energy_drift"] = quantum_energy - reference
                group.attrs["hamiltonian_hermiticity_residual"] = float(
                    np.max(np.abs(
                        state.matrices.hamiltonian - state.matrices.hamiltonian.conj().T
                    ))
                )
            if diagnostics:
                for key, value in diagnostics.items():
                    if value is None:
                        continue
                    array = np.asarray(value)
                    if array.ndim:
                        group.create_dataset(key, data=array)
                    else:
                        group.attrs[key] = array.item()
            if state.events:
                group.attrs["events_json"] = json.dumps(state.events, default=_json_default)
            if self.replay_window is None:
                steps.attrs["committed_through"] = max(
                    int(steps.attrs.get("committed_through", -1)), state.step
                )
            handle.flush()
        if self.xyz_root is not None and self.replay_window is None:
            _write_xyz_step(
                self.xyz_root,
                self.atoms,
                name,
                state.quantum_time,
                np.asarray([trajectory.positions for trajectory in trajectories]),
                [trajectory.label for trajectory in trajectories],
                [trajectory.identifier for trajectory in trajectories],
                [trajectory.state for trajectory in trajectories],
            )


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _safe_filename(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    ).strip(".")
    return safe or "tbf"


def _write_xyz_step(
    output: Path,
    atoms: tuple[str, ...],
    step_name: str,
    time_au: float,
    positions_bohr: np.ndarray,
    labels: list[str],
    identifiers: list[str],
    states: list[int],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"step-{step_name}"
    temporary = output / f".step-{step_name}-{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for index, (positions, label, identifier, state) in enumerate(
            zip(positions_bohr, labels, identifiers, states, strict=True)
        ):
            if positions.shape != (len(atoms), 3):
                raise ValueError(
                    f"XYZ positions must have shape ({len(atoms)}, 3), "
                    f"got {positions.shape}"
                )
            name = f"tbf-{index:04d}-{_safe_filename(label)}.xyz"
            lines = [
                str(len(atoms)),
                (
                    f"step={int(step_name)} time_au={float(time_au):.16g} "
                    f"tbf_index={index} label={label} id={identifier} state={int(state)}"
                ),
            ]
            for atom, coordinate in zip(atoms, positions / BOHR_PER_ANGSTROM, strict=True):
                lines.append(
                    f"{atom:<3s} {coordinate[0]: .12f} "
                    f"{coordinate[1]: .12f} {coordinate[2]: .12f}"
                )
            (temporary / name).write_text("\n".join(lines) + "\n")
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def export_xyz_history(
    history: str | Path,
    output_directory: str | Path,
) -> Path:
    """Export every committed TBF geometry as an Angstrom XYZ file."""
    history = Path(history)
    if history.is_dir():
        history = history / "simulation.h5"
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(history, "r") as handle:
        if "steps" not in handle:
            raise ValueError(f"{history} is not an aims_berry history")
        if "atoms" in handle:
            atoms = tuple(_text(value) for value in handle["atoms"][:])
        else:
            config = json.loads(handle.attrs["config_json"])
            atoms = read_xyz(config["geometry"], config.get("geometry_units", "angstrom"))[0]
        steps = handle["steps"]
        committed = int(steps.attrs.get("committed_through", -1))
        names = sorted(
            (name for name in steps if int(name) <= committed), key=int
        )
        expected = {f"step-{name}" for name in names}
        for child in output.glob("step-*"):
            if child.is_dir() and child.name not in expected:
                shutil.rmtree(child)
        for name in names:
            group = steps[name]
            _write_xyz_step(
                output,
                atoms,
                name,
                float(group.attrs["time"]),
                group["positions"][:],
                [_text(value) for value in group["labels"][:]],
                [_text(value) for value in group["ids"][:]],
                group["states"][:].astype(int).tolist(),
            )
    return output


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


class CheckpointManager:
    def __init__(self, root: str | Path, keep: int = 2) -> None:
        self.root = Path(root)
        self.keep = max(1, int(keep))
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def current(self) -> Path:
        return self.root / "current"

    def save(
        self,
        state: SimulationState,
        queue: TaskQueue,
        rng: np.random.Generator,
        config: SimulationConfig,
        provider: ElectronicStructureProvider,
        runtime: dict[str, Any] | None = None,
    ) -> Path:
        temporary = self.root / f".tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        provider_directory = temporary / "provider"
        provider_directory.mkdir()
        provider_metadata = provider.dump_state(provider_directory)
        metadata = {
            "schema_version": 3,
            "version": __version__,
            "quantum_time": state.quantum_time,
            "step": state.step,
            "events": state.events,
            "queue": queue.to_dict(),
            "rng_state": rng.bit_generator.state,
            "config": config.to_dict(),
            "provider": provider_metadata,
            "runtime": runtime or {},
            "trajectories": [
                {
                    "state": t.state, "time": t.time, "label": t.label,
                    "identifier": t.identifier, "parent_id": t.parent_id,
                    "generation": t.generation, "phase": t.phase,
                    "last_spawn_time": t.last_spawn_time, "spawn_count": t.spawn_count,
                    "wavefunction_id": t.previous_wavefunction.identifier if t.previous_wavefunction else None,
                }
                for t in state.trajectories
            ],
        }
        (temporary / "checkpoint.json").write_text(json.dumps(metadata, default=_json_default, indent=2, sort_keys=True))
        arrays: dict[str, Any] = {"amplitudes": state.amplitudes}
        for index, trajectory in enumerate(state.trajectories):
            for field in ("positions", "momenta", "widths", "masses"):
                arrays[f"t{index}_{field}"] = getattr(trajectory, field)
            if trajectory.electronic is not None:
                arrays[f"t{index}_energies"] = trajectory.electronic.energies
                if trajectory.electronic.gradients is not None:
                    arrays[f"t{index}_gradients"] = trajectory.electronic.gradients
                    arrays[f"t{index}_gradient_mask"] = trajectory.electronic.gradient_mask
                if trajectory.electronic.nacs is not None:
                    arrays[f"t{index}_nacs"] = trajectory.electronic.nacs
                    arrays[f"t{index}_nac_mask"] = trajectory.electronic.nac_mask
                if trajectory.electronic.state_overlaps is not None:
                    arrays[f"t{index}_state_overlaps"] = trajectory.electronic.state_overlaps
        if state.matrices is not None:
            arrays.update(H=state.matrices.hamiltonian, S=state.matrices.overlap, Sdot=state.matrices.sdot)
        np.savez_compressed(temporary / "state.npz", **arrays)
        previous = self.root / "previous"
        if previous.exists():
            shutil.rmtree(previous)
        if self.current.exists():
            os.replace(self.current, previous)
        os.replace(temporary, self.current)
        if self.keep == 1 and previous.exists():
            shutil.rmtree(previous)
        return self.current

    def load(self, provider: ElectronicStructureProvider) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if not self.current.exists():
            raise FileNotFoundError(f"no checkpoint at {self.current}")
        metadata = json.loads((self.current / "checkpoint.json").read_text())
        archive = np.load(self.current / "state.npz", allow_pickle=False)
        arrays = {name: archive[name] for name in archive.files}
        provider.load_state(self.current / "provider", metadata.get("provider", {}))
        return metadata, arrays
