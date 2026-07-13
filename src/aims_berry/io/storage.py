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
from ..config import SimulationConfig
from ..core import SimulationState
from ..electronic.base import ElectronicStructureProvider
from ..tasks import TaskQueue


class HDF5Writer:
    def __init__(self, path: str | Path, config: SimulationConfig, *, restart: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if restart else "w"
        with h5py.File(self.path, mode, libver="latest") as handle:
            handle.attrs["schema_version"] = 1
            handle.attrs["aims_berry_version"] = __version__
            handle.attrs["units"] = "atomic"
            handle.attrs["config_json"] = json.dumps(config.to_dict(), sort_keys=True)
            handle.require_group("steps")

    def write_step(self, state: SimulationState, num_states: int) -> None:
        with h5py.File(self.path, "a", libver="latest") as handle:
            steps = handle["steps"]
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
            energies = np.full((len(trajectories), num_states), np.nan)
            gradients = np.full((len(trajectories), num_states, trajectories[0].positions.shape[0], 3), np.nan)
            nacs = np.full(
                (len(trajectories), num_states, num_states, trajectories[0].positions.shape[0], 3),
                np.nan + 0j,
            )
            state_overlaps = np.full(
                (len(trajectories), num_states, num_states), np.nan + 0j
            )
            for index, trajectory in enumerate(trajectories):
                if trajectory.electronic is not None:
                    energies[index] = trajectory.electronic.energies
                    if trajectory.electronic.gradients is not None:
                        gradients[index] = trajectory.electronic.gradients
                    if trajectory.electronic.nacs is not None:
                        nacs[index] = trajectory.electronic.nacs
                    if trajectory.electronic.state_overlaps is not None:
                        state_overlaps[index] = trajectory.electronic.state_overlaps
            group.create_dataset("energies", data=energies)
            group.create_dataset("gradients", data=gradients, compression="gzip")
            group.create_dataset("nacs", data=nacs, compression="gzip")
            group.create_dataset("state_overlaps", data=state_overlaps)
            if state.matrices is not None:
                group.create_dataset("H", data=state.matrices.hamiltonian)
                group.create_dataset("S", data=state.matrices.overlap)
                group.create_dataset("Sdot", data=state.matrices.sdot)
            if state.events:
                group.attrs["events_json"] = json.dumps(state.events, default=_json_default)
            handle.flush()


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
            "schema_version": 1,
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
                if trajectory.electronic.nacs is not None:
                    arrays[f"t{index}_nacs"] = trajectory.electronic.nacs
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
