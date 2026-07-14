"""Task-driven AIMS simulation orchestration."""

from __future__ import annotations

import copy
import dataclasses
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig, config_from_dict
from .core import MatrixSet, SimulationState, TrajectoryBasisFunction
from .dynamics.classical import berry_boris_step, velocity_verlet
from .dynamics.hamiltonian import BerryExactHamiltonian, SaddlePointHamiltonian
from .dynamics.quantum import cayley_step, metric_norm, normalize, rk45_step
from .electronic.base import (
    ElectronicProperties,
    ElectronicStructureError,
    ElectronicStructureProvider,
    ElectronicStructureRequest,
    ElectronicStructureResult,
    ProviderCapabilityError,
    WavefunctionState,
)
from .geometry import atomic_numbers, masses_and_widths, read_momenta, read_xyz, sample_wigner
from .gauge.berry import GaugeTracker, transform_nacs
from .io.storage import CheckpointManager, HDF5Writer
from .spawning import (
    SpawnCandidate,
    SpawnMonitor,
    make_coupling_optimized_child,
    make_child,
    overlaps_existing,
    prune_by_overlap,
    projected_coupling,
    trajectory_populations,
)
from .tasks import SerialExecutor, Task, TaskKind, TaskQueue


@dataclasses.dataclass(frozen=True)
class SimulationResult:
    run_directory: Path
    history: Path
    checkpoint: Path
    state: SimulationState
    events: tuple[dict[str, Any], ...]


def _array_payload(value: np.ndarray | None) -> dict[str, Any] | None:
    if value is None:
        return None
    array = np.asarray(value)
    payload: dict[str, Any] = {"real": array.real.tolist(), "shape": list(array.shape)}
    if np.iscomplexobj(array):
        payload["imag"] = array.imag.tolist()
    return payload


def _array_from_payload(payload: dict[str, Any] | None) -> np.ndarray | None:
    if payload is None:
        return None
    array = np.asarray(payload["real"])
    if "imag" in payload:
        array = array + 1j * np.asarray(payload["imag"])
    return array.reshape(payload["shape"])


def _state_payload(state: SimulationState) -> dict[str, Any]:
    trajectories = []
    for trajectory in state.trajectories:
        electronic = trajectory.electronic
        trajectories.append({
            "positions": _array_payload(trajectory.positions),
            "momenta": _array_payload(trajectory.momenta),
            "widths": _array_payload(trajectory.widths),
            "masses": _array_payload(trajectory.masses),
            "state": trajectory.state,
            "time": trajectory.time,
            "label": trajectory.label,
            "identifier": trajectory.identifier,
            "parent_id": trajectory.parent_id,
            "generation": trajectory.generation,
            "phase": trajectory.phase,
            "last_spawn_time": trajectory.last_spawn_time,
            "spawn_count": trajectory.spawn_count,
            "wavefunction_id": (
                trajectory.previous_wavefunction.identifier
                if trajectory.previous_wavefunction is not None else None
            ),
            "electronic": None if electronic is None else {
                "energies": _array_payload(electronic.energies),
                "gradients": _array_payload(electronic.gradients),
                "nacs": _array_payload(electronic.nacs),
                "state_overlaps": _array_payload(electronic.state_overlaps),
            },
        })
    matrices = None if state.matrices is None else {
        "S": _array_payload(state.matrices.overlap),
        "H": _array_payload(state.matrices.hamiltonian),
        "Sdot": _array_payload(state.matrices.sdot),
    }
    return {
        "trajectories": trajectories,
        "amplitudes": _array_payload(state.amplitudes),
        "quantum_time": state.quantum_time,
        "step": state.step,
        "matrices": matrices,
        "events": copy.deepcopy(state.events),
    }


def _state_from_payload(payload: dict[str, Any]) -> SimulationState:
    trajectories = []
    for item in payload["trajectories"]:
        electronic_payload = item["electronic"]
        electronic = None
        if electronic_payload is not None:
            electronic = ElectronicStructureResult(
                energies=_array_from_payload(electronic_payload["energies"]),
                gradients=_array_from_payload(electronic_payload["gradients"]),
                nacs=_array_from_payload(electronic_payload["nacs"]),
                state_overlaps=_array_from_payload(electronic_payload["state_overlaps"]),
            )
        wavefunction_id = item["wavefunction_id"]
        trajectories.append(TrajectoryBasisFunction(
            positions=_array_from_payload(item["positions"]),
            momenta=_array_from_payload(item["momenta"]),
            widths=_array_from_payload(item["widths"]),
            masses=_array_from_payload(item["masses"]),
            state=item["state"],
            time=item["time"],
            label=item["label"],
            identifier=item["identifier"],
            parent_id=item["parent_id"],
            generation=item["generation"],
            phase=item["phase"],
            previous_wavefunction=(
                WavefunctionState(wavefunction_id) if wavefunction_id is not None else None
            ),
            electronic=electronic,
            last_spawn_time=item["last_spawn_time"],
            spawn_count=item["spawn_count"],
        ))
    matrices_payload = payload["matrices"]
    matrices = None if matrices_payload is None else MatrixSet(
        _array_from_payload(matrices_payload["S"]),
        _array_from_payload(matrices_payload["H"]),
        _array_from_payload(matrices_payload["Sdot"]),
    )
    return SimulationState(
        trajectories=trajectories,
        amplitudes=_array_from_payload(payload["amplitudes"]),
        quantum_time=payload["quantum_time"],
        step=payload["step"],
        matrices=matrices,
        events=copy.deepcopy(payload["events"]),
    )


@dataclasses.dataclass
class SpawnReplaySnapshot:
    key: tuple[str, int]
    state: SimulationState
    queue: dict[str, Any]
    rng_state: dict[str, Any]
    monitor: dict[str, Any]
    centroid_wavefunctions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": [self.key[0], self.key[1]],
            "state": _state_payload(self.state),
            "queue": self.queue,
            "rng_state": self.rng_state,
            "monitor": self.monitor,
            "centroid_wavefunctions": self.centroid_wavefunctions,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpawnReplaySnapshot":
        return cls(
            key=(str(payload["key"][0]), int(payload["key"][1])),
            state=_state_from_payload(payload["state"]),
            queue=payload["queue"],
            rng_state=payload["rng_state"],
            monitor=payload["monitor"],
            centroid_wavefunctions=dict(payload.get("centroid_wavefunctions", {})),
        )


def _provider_from_config(config: SimulationConfig) -> ElectronicStructureProvider:
    if config.provider == "pyscf":
        from .electronic.pyscf import from_config
        return from_config(config)
    if config.provider in {"berry", "berry_2d"}:
        from .models import BerryModel2DParallelTransport
        options = config.provider_option_dict()
        return BerryModel2DParallelTransport(
            energy=float(options.get("energy", 0.02)),
            sharpness=float(options.get("sharpness", 3.0)),
            phase_gradient=float(options.get("phase_gradient", 5.0)),
        )
    if config.provider in {"python", "custom"}:
        factory_path = config.provider_option_dict().get("factory")
        if not isinstance(factory_path, str) or ":" not in factory_path:
            raise ValueError("custom provider requires `provider_option factory module:object`")
        module_name, object_name = factory_path.split(":", 1)
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            candidate = config.source.parent / f"{module_name.replace('.', '/')}.py" if config.source else None
            if candidate is None or not candidate.is_file():
                raise
            spec = importlib.util.spec_from_file_location(f"aims_berry_user_{module_name}", candidate)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load custom provider module {candidate}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        factory = getattr(module, object_name)
        return factory(config)
    raise ValueError(f"unknown provider {config.provider!r}")


class SimulationRunner:
    def __init__(self, config: SimulationConfig, provider: ElectronicStructureProvider, *, restart: bool = False):
        self.config = config
        self.provider = provider
        self.is_restart = restart
        self.atoms, initial_positions = read_xyz(config.geometry, config.geometry_units)
        self.atomic_numbers = atomic_numbers(self.atoms)
        masses, widths = masses_and_widths(self.atoms, config.gaussian_widths)
        self.masses = masses.reshape(initial_positions.shape)
        self.widths = widths.reshape(initial_positions.shape)
        self.rng = np.random.default_rng(config.random_seed)
        self.queue = TaskQueue()
        self.executor = SerialExecutor()
        self.monitor = SpawnMonitor(config.spawn_threshold)
        self.gauge_tracker = GaugeTracker()
        self.centroid_cache: dict[bytes, ElectronicStructureResult] = {}
        self.centroid_previous: dict[str, WavefunctionState] = {}
        self.active_spawn_snapshot: SpawnReplaySnapshot | None = None
        self.replay_suppressed: dict[tuple[str, int], float] = {}
        self.run_directory = config.run_directory
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoints = CheckpointManager(self.run_directory / "checkpoint", config.checkpoint_keep)
        if restart:
            self.state = self._restore_state()
        else:
            momenta = np.zeros_like(initial_positions)
            if config.initial_condition == "wigner":
                initial_positions, momenta = sample_wigner(
                    initial_positions, masses, config.hessian, self.rng, config.temperature
                )
            elif config.momenta is not None:
                momenta = read_momenta(config.momenta, len(self.atoms))
            trajectory = TrajectoryBasisFunction(
                positions=initial_positions,
                momenta=momenta,
                widths=self.widths,
                masses=self.masses,
                state=config.initial_state,
                identifier=f"tbf-{config.random_seed:016x}-0000",
            )
            self.state = SimulationState([trajectory], np.ones(1, dtype=np.complex128))
        self.writer = HDF5Writer(self.run_directory / "simulation.h5", config, restart=restart)
        self._validate_capabilities()

    def _validate_capabilities(self) -> None:
        if not self.provider.capabilities.energies or not self.provider.capabilities.gradients:
            raise ProviderCapabilityError("dynamics requires energies and gradients")
        mode = self._coupling_mode()
        if self.config.num_states > 1 and mode == "nac" and not self.provider.capabilities.nacs:
            raise ProviderCapabilityError("NAC coupling requested but unavailable")
        if self.config.num_states > 1 and mode == "npi" and not self.provider.capabilities.npi_tdc:
            raise ProviderCapabilityError("NPI coupling requested but provider overlaps are not certified")
        if self.config.spawn_strategy == "coupling_optimized":
            try:
                import cma  # noqa: F401
            except ImportError as exc:
                raise ProviderCapabilityError(
                    "coupling_optimized spawning requires `pip install aims-berry[optimize]`"
                ) from exc

    def _coupling_mode(self) -> str:
        if self.config.coupling_mode != "auto":
            return self.config.coupling_mode
        return "npi" if self.provider.capabilities.npi_tdc else "nac"

    def _request(
        self,
        geometry: np.ndarray,
        active_state: int,
        time: float,
        previous=None,
        *,
        gradients: bool = True,
    ) -> ElectronicStructureRequest:
        properties = {ElectronicProperties.ENERGIES}
        if gradients:
            properties.add(ElectronicProperties.GRADIENTS)
        if self.config.num_states > 1:
            properties.add(ElectronicProperties.STATE_OVERLAPS if self._coupling_mode() == "npi" else ElectronicProperties.NACS)
        return ElectronicStructureRequest(
            atoms=self.atoms,
            atomic_numbers=self.atomic_numbers,
            geometry=np.asarray(geometry, dtype=float),
            states=tuple(range(self.config.num_states)),
            properties=frozenset(properties),
            active_state=active_state,
            time=time,
            previous=previous,
        )

    def _evaluate(
        self,
        geometry: np.ndarray,
        active_state: int,
        time: float,
        previous=None,
        *,
        gradients: bool = True,
    ) -> ElectronicStructureResult:
        request = self._request(geometry, active_state, time, previous, gradients=gradients)
        last_error: BaseException | None = None
        for _attempt in range(self.config.electronic_retries + 1):
            try:
                result = self.provider.evaluate(request).validate(request)
                if result.state_overlaps is not None:
                    transform = self.gauge_tracker.align(result.state_overlaps)
                    order = transform.permutation
                    result.energies = result.energies[order]
                    if result.gradients is not None:
                        result.gradients = result.gradients[order]
                    if result.nacs is not None:
                        reordered = result.nacs[order][:, order]
                        result.nacs = transform_nacs(reordered, transform.unitary)
                    result.state_overlaps = transform.aligned_overlap
                    result.metadata["gauge_permutation"] = order.tolist()
                    result.metadata["gauge_unitary"] = transform.unitary
                    result.validate(request)
                return result
            except ElectronicStructureError as exc:
                last_error = exc
                if not exc.retryable:
                    raise
        assert last_error is not None
        raise last_error

    def _evaluate_trajectory(self, index: int) -> None:
        trajectory = self.state.trajectories[index]
        result = self._evaluate(
            trajectory.positions, trajectory.state, trajectory.time, trajectory.previous_wavefunction
        )
        trajectory.electronic = result
        trajectory.previous_wavefunction = result.wavefunction

    @staticmethod
    def _centroid_pair_key(
        left: TrajectoryBasisFunction, right: TrajectoryBasisFunction
    ) -> str:
        return f"{left.identifier}:{right.identifier}"

    def _centroid_key(
        self,
        left: TrajectoryBasisFunction,
        right: TrajectoryBasisFunction,
        geometry: np.ndarray,
    ) -> bytes:
        pair = self._centroid_pair_key(left, right).encode()
        return pair + b"\0" + np.round(geometry, 12).tobytes()

    def _evaluate_centroid(
        self,
        left: TrajectoryBasisFunction,
        right: TrajectoryBasisFunction,
        geometry: np.ndarray,
    ) -> ElectronicStructureResult:
        key = self._centroid_key(left, right, geometry)
        if key not in self.centroid_cache:
            pair = self._centroid_pair_key(left, right)
            result = self._evaluate(
                geometry,
                right.state,
                self.state.quantum_time,
                self.centroid_previous.get(pair),
                gradients=False,
            )
            self.centroid_cache[key] = result
            if result.wavefunction is not None:
                self.centroid_previous[pair] = result.wavefunction
        return self.centroid_cache[key]

    def _build_matrices(self) -> MatrixSet:
        velocities = [trajectory.momenta / trajectory.masses for trajectory in self.state.trajectories]
        forces = [
            -trajectory.electronic.gradients[trajectory.state]
            for trajectory in self.state.trajectories
        ]
        assembler = (
            BerryExactHamiltonian(self.provider)
            if self.provider.__class__.__name__ == "BerryModel2DParallelTransport"
            else SaddlePointHamiltonian(self._coupling_mode())
        )
        matrices = assembler.build(
            self.state.trajectories, self._evaluate_centroid, velocities, forces, self.config.time_step
        )
        self.state.matrices = matrices
        return matrices

    def _propagate_trajectory(self, index: int, dt: float | None = None) -> None:
        trajectory = self.state.trajectories[index]
        dt = self.config.time_step if dt is None else float(dt)
        endpoint: ElectronicStructureResult | None = None
        if all(hasattr(self.provider, name) for name in ("connection", "curvature", "scalar_gradient")):
            trajectory.positions, trajectory.momenta = berry_boris_step(
                trajectory.positions, trajectory.momenta, trajectory.masses, trajectory.state, dt,
                self.provider.connection, self.provider.curvature, self.provider.scalar_gradient,
            )
        else:
            gradient0 = trajectory.electronic.gradients[trajectory.state]

            def gradient_at(positions: np.ndarray) -> np.ndarray:
                nonlocal endpoint
                endpoint = self._evaluate(
                    positions,
                    trajectory.state,
                    trajectory.time + dt,
                    trajectory.previous_wavefunction,
                )
                return endpoint.gradients[trajectory.state]

            trajectory.positions, trajectory.momenta, _ = velocity_verlet(
                trajectory.positions, trajectory.momenta, trajectory.masses, dt, gradient0, gradient_at
            )
        trajectory.time += dt
        trajectory.electronic = endpoint
        if endpoint is not None:
            trajectory.previous_wavefunction = endpoint.wavefunction
        self.state.matrices = None

    def _capture_spawn_snapshot(self, key: tuple[str, int]) -> None:
        self.active_spawn_snapshot = SpawnReplaySnapshot(
            key=key,
            state=_state_from_payload(_state_payload(self.state)),
            queue=copy.deepcopy(self.queue.to_dict()),
            rng_state=copy.deepcopy(self.rng.bit_generator.state),
            monitor=copy.deepcopy(self.monitor.to_dict()),
            centroid_wavefunctions={
                name: wavefunction.identifier
                for name, wavefunction in self.centroid_previous.items()
            },
        )

    def _restore_spawn_snapshot(self, snapshot: SpawnReplaySnapshot) -> None:
        self.state = _state_from_payload(_state_payload(snapshot.state))
        self.queue = TaskQueue.from_dict(copy.deepcopy(snapshot.queue))
        self.rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)
        self.monitor = SpawnMonitor.from_dict(copy.deepcopy(snapshot.monitor))
        self.centroid_previous = {
            key: WavefunctionState(identifier)
            for key, identifier in snapshot.centroid_wavefunctions.items()
        }
        self.centroid_cache.clear()

    def _observe_spawning(self) -> list[SpawnCandidate]:
        if self.config.num_states < 2 or len(self.state.trajectories) >= self.config.max_trajectories:
            return []
        contributions = trajectory_populations(self.state)
        emitted: list[SpawnCandidate] = []
        for index, trajectory in enumerate(self.state.trajectories):
            if trajectory.electronic is None or trajectory.electronic.nacs is None:
                continue
            for target in range(self.config.num_states):
                if target == trajectory.state:
                    continue
                key = (trajectory.identifier, target)
                frontier = self.replay_suppressed.get(key)
                if frontier is not None:
                    if trajectory.time <= frontier + 1.0e-12:
                        continue
                    del self.replay_suppressed[key]
                if self.active_spawn_snapshot is not None:
                    if key != self.active_spawn_snapshot.key:
                        continue
                else:
                    if contributions[index] < self.config.population_to_spawn:
                        continue
                    if trajectory.time - trajectory.last_spawn_time < self.config.spawn_cooldown:
                        continue
                gap = abs(trajectory.electronic.energies[target] - trajectory.electronic.energies[trajectory.state])
                if gap > self.config.max_energy_gap:
                    continue
                nac = trajectory.electronic.nacs[trajectory.state, target]
                candidate = SpawnCandidate(
                    index, target, projected_coupling(trajectory, nac), trajectory.time,
                    trajectory.positions.copy(), trajectory.momenta.copy(), nac.copy(),
                    trajectory.electronic.energies.copy(),
                    parent_id=trajectory.identifier,
                )
                if (
                    self.active_spawn_snapshot is None
                    and candidate.coupling >= self.config.spawn_threshold
                    and SpawnMonitor.key(candidate) not in self.monitor.pending
                ):
                    self._capture_spawn_snapshot(key)
                completed = self.monitor.observe(candidate)
                if completed is not None:
                    emitted.append(completed)
        return emitted

    def _spawn(self, candidate: SpawnCandidate) -> None:
        if len(self.state.trajectories) >= self.config.max_trajectories:
            return
        parent_index = next(
            (
                index for index, trajectory in enumerate(self.state.trajectories)
                if trajectory.identifier == candidate.parent_id
            ),
            candidate.parent_index,
        )
        parent = self.state.trajectories[parent_index]
        child = (
            make_coupling_optimized_child(candidate, parent, self.config.random_seed + parent.spawn_count)
            if self.config.spawn_strategy == "coupling_optimized"
            else make_child(candidate, parent)
        )
        if child is None:
            self.state.events.append({"kind": "failed_spawn", "time": candidate.time, "reason": "energy_shell"})
            self.active_spawn_snapshot = None
            return
        if overlaps_existing(child, self.state.trajectories, self.config.spawn_overlap_max):
            self.state.events.append({"kind": "failed_spawn", "time": candidate.time, "reason": "overlap"})
            self.active_spawn_snapshot = None
            return
        # Build the child's missing history back to the threshold-entry time.
        entry_time = candidate.entry_time if candidate.entry_time is not None else candidate.time
        while child.time - 1.0e-12 > entry_time:
            step = -min(self.config.time_step, child.time - entry_time)
            child.electronic = self._evaluate(child.positions, child.state, child.time, child.previous_wavefunction)
            child.previous_wavefunction = child.electronic.wavefunction
            self.state.trajectories.append(child)
            try:
                self._propagate_trajectory(len(self.state.trajectories) - 1, step)
            finally:
                self.state.trajectories.pop()

        snapshot = self.active_spawn_snapshot
        if snapshot is None:
            raise RuntimeError("spawn replay snapshot is missing")
        frontier_time = self.state.quantum_time
        frontier_step = self.state.step
        replay_key = snapshot.key
        self._restore_spawn_snapshot(snapshot)
        self.active_spawn_snapshot = None
        restored_parent = next(
            trajectory for trajectory in self.state.trajectories
            if trajectory.identifier == replay_key[0]
        )
        restored_parent.spawn_count += 1
        restored_parent.last_spawn_time = candidate.time
        index = self.state.add_trajectory(child, amplitude=0j)
        if child.electronic is None:
            self._evaluate_trajectory(index)
        self.replay_suppressed[replay_key] = frontier_time
        self.state.events.append({
            "kind": "spawn", "time": candidate.time, "parent": parent.label,
            "child": child.label, "target_state": child.state, "coupling": candidate.coupling,
            "index": index,
            "threshold_entry_time": entry_time,
            "replay_frontier_time": frontier_time,
        })
        self.state.matrices = None
        self._build_matrices()
        self.writer.write_step(self.state, self.config.num_states)
        self._checkpoint()
        self.propagate(stop_after_step=frontier_step)

    def _checkpoint(self) -> None:
        references = {
            trajectory.previous_wavefunction.identifier
            for trajectory in self.state.trajectories
            if trajectory.previous_wavefunction is not None
        }
        references.update(value.identifier for value in self.centroid_previous.values())
        if self.active_spawn_snapshot is not None:
            references.update(
                trajectory.previous_wavefunction.identifier
                for trajectory in self.active_spawn_snapshot.state.trajectories
                if trajectory.previous_wavefunction is not None
            )
            references.update(self.active_spawn_snapshot.centroid_wavefunctions.values())
        set_references = getattr(self.provider, "set_checkpoint_references", None)
        if callable(set_references):
            set_references(frozenset(references))
        self.checkpoints.save(
            self.state, self.queue, self.rng, self.config, self.provider,
            runtime={
                "spawn_monitor": self.monitor.to_dict(),
                "centroid_wavefunctions": {
                    key: value.identifier for key, value in self.centroid_previous.items()
                },
                "active_spawn_snapshot": (
                    self.active_spawn_snapshot.to_dict()
                    if self.active_spawn_snapshot is not None else None
                ),
                "replay_suppressed": [
                    [key[0], key[1], frontier]
                    for key, frontier in self.replay_suppressed.items()
                ],
            },
        )

    def _execute(self, task: Task, action) -> None:
        try:
            if not task.dependencies <= self.queue.completed:
                missing = sorted(task.dependencies - self.queue.completed)
                raise RuntimeError(f"task {task.identifier} has incomplete dependencies: {missing}")
            self.executor.execute(task, action)
            self.queue.complete(task)
            self.state.events.append({"kind": "task", "task": task.kind.value, "id": task.identifier, "time": self.state.quantum_time})
            self._checkpoint()
        except BaseException as exc:
            self.queue.fail(task, exc)
            self._checkpoint()
            raise

    def _restore_state(self) -> SimulationState:
        metadata, arrays = self.checkpoints.load(self.provider)
        self.queue = TaskQueue.from_dict(metadata["queue"])
        self.rng.bit_generator.state = metadata["rng_state"]
        if metadata.get("runtime", {}).get("spawn_monitor"):
            self.monitor = SpawnMonitor.from_dict(metadata["runtime"]["spawn_monitor"])
        self.centroid_previous = {
            key: WavefunctionState(identifier)
            for key, identifier in metadata.get("runtime", {})
            .get("centroid_wavefunctions", {})
            .items()
        }
        active_spawn = metadata.get("runtime", {}).get("active_spawn_snapshot")
        self.active_spawn_snapshot = (
            SpawnReplaySnapshot.from_dict(active_spawn) if active_spawn is not None else None
        )
        self.replay_suppressed = {
            (str(parent), int(target)): float(frontier)
            for parent, target, frontier in metadata.get("runtime", {}).get(
                "replay_suppressed", []
            )
        }
        trajectories = []
        for index, item in enumerate(metadata["trajectories"]):
            trajectory = TrajectoryBasisFunction(
                positions=arrays[f"t{index}_positions"], momenta=arrays[f"t{index}_momenta"],
                widths=arrays[f"t{index}_widths"], masses=arrays[f"t{index}_masses"],
                state=item["state"], time=item["time"], label=item["label"],
                identifier=item["identifier"], parent_id=item["parent_id"], generation=item["generation"],
                phase=item["phase"], last_spawn_time=item["last_spawn_time"], spawn_count=item["spawn_count"],
                previous_wavefunction=WavefunctionState(item["wavefunction_id"]) if item.get("wavefunction_id") else None,
            )
            if f"t{index}_energies" in arrays:
                trajectory.electronic = ElectronicStructureResult(
                    energies=arrays[f"t{index}_energies"],
                    gradients=arrays.get(f"t{index}_gradients"), nacs=arrays.get(f"t{index}_nacs"),
                    state_overlaps=arrays.get(f"t{index}_state_overlaps"),
                )
            trajectories.append(trajectory)
        matrices = None
        if all(key in arrays for key in ("H", "S", "Sdot")):
            matrices = MatrixSet(arrays["S"], arrays["H"], arrays["Sdot"])
        return SimulationState(
            trajectories, arrays["amplitudes"], metadata["quantum_time"], metadata["step"], matrices,
            metadata.get("events", []),
        )

    def propagate(self, *, stop_after_step: int | None = None) -> SimulationResult:
        if not self.is_restart and self.state.step == 0:
            for index in range(len(self.state.trajectories)):
                self._evaluate_trajectory(index)
            self._build_matrices()
            self.writer.write_step(self.state, self.config.num_states)
            self._checkpoint()
        target_step = self.config.nsteps
        if stop_after_step is not None:
            target_step = min(target_step, max(self.state.step, int(stop_after_step)))
        while self.state.step < target_step:
            self.centroid_cache.clear()
            for index, trajectory in enumerate(self.state.trajectories):
                if trajectory.electronic is None:
                    self._evaluate_trajectory(index)

            start_matrices = self.state.matrices or self._build_matrices()
            removed = prune_by_overlap(self.state, self.config.overlap_threshold)
            if removed:
                self.state.events.append({
                    "kind": "prune", "time": self.state.quantum_time, "indices": removed,
                })
                start_matrices = self._build_matrices()

            for candidate in self._observe_spawning():
                task = self.queue.add(TaskKind.SPAWN, candidate.time, f"spawn:{candidate.parent_index}:{candidate.target_state}:{candidate.time}")
                self._execute(task, lambda current, item=candidate: self._spawn(item))
                start_matrices = self._build_matrices()

            propagation_tasks = []
            for index, trajectory in enumerate(list(self.state.trajectories)):
                task = self.queue.add(
                    TaskKind.TRAJECTORY,
                    trajectory.time,
                    f"prop:{self.state.step}:{trajectory.identifier}",
                )
                propagation_tasks.append(task)
                self._execute(task, lambda current, idx=index: self._propagate_trajectory(idx))
            self.state.quantum_time += self.config.time_step
            self.state.step += 1
            self.centroid_cache.clear()
            for index, trajectory in enumerate(self.state.trajectories):
                if trajectory.electronic is None:
                    self._evaluate_trajectory(index)
            end_matrices = self._build_matrices()

            quantum = self.queue.add(
                TaskKind.QUANTUM,
                self.state.quantum_time,
                f"quantum:{self.state.step - 1}",
                dependencies=[task.identifier for task in propagation_tasks],
            )

            def quantum_action(_task: Task) -> None:
                before = metric_norm(self.state.amplitudes, start_matrices.overlap)
                integrator = cayley_step if self.config.quantum_integrator == "cayley" else rk45_step
                propagated = integrator(
                    self.state.amplitudes,
                    start_matrices,
                    self.config.time_step,
                    self.config.regularization_threshold,
                    end_matrices,
                )
                after = metric_norm(propagated, end_matrices.overlap)
                if abs(after - before) > self.config.norm_tolerance:
                    raise RuntimeError(
                        f"quantum norm violation: before={before}, after={after}"
                    )
                self.state.amplitudes = normalize(propagated, end_matrices.overlap)
                self.state.matrices = end_matrices

            self._execute(quantum, quantum_action)
            if self.state.step % self.config.output_every == 0 or self.state.step == self.config.nsteps:
                output = self.queue.add(
                    TaskKind.OUTPUT,
                    self.state.quantum_time,
                    f"output:{self.state.step}",
                    dependencies=[quantum.identifier],
                )
                self._execute(output, lambda current: self.writer.write_step(self.state, self.config.num_states))
        if self.state.step < self.config.nsteps:
            self._checkpoint()
            return SimulationResult(
                self.run_directory, self.run_directory / "simulation.h5", self.checkpoints.current,
                self.state, tuple(self.state.events),
            )
        spawned_at_end = False
        for candidate in self.monitor.flush():
            before = len(self.state.trajectories)
            self._spawn(candidate)
            spawned_at_end |= len(self.state.trajectories) > before
        if spawned_at_end:
            for index in range(len(self.state.trajectories)):
                self._evaluate_trajectory(index)
            self._build_matrices()
            self.writer.write_step(self.state, self.config.num_states)
        self._checkpoint()
        return SimulationResult(
            self.run_directory, self.run_directory / "simulation.h5", self.checkpoints.current,
            self.state, tuple(self.state.events),
        )


def run(
    config: SimulationConfig,
    provider: ElectronicStructureProvider | None = None,
    *,
    restart: bool = False,
) -> SimulationResult:
    provider = provider or _provider_from_config(config)
    return SimulationRunner(config, provider, restart=restart).propagate()


def restart_from_checkpoint(
    checkpoint_directory: str | Path,
    provider: ElectronicStructureProvider | None = None,
) -> SimulationResult:
    checkpoint_directory = Path(checkpoint_directory).resolve()
    current = checkpoint_directory if checkpoint_directory.name == "current" else checkpoint_directory / "current"
    metadata = json.loads((current / "checkpoint.json").read_text())
    config = config_from_dict(metadata["config"])
    config = dataclasses.replace(config, run_directory=current.parent.parent)
    return run(config, provider, restart=True)
