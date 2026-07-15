"""Task-driven AIMS simulation orchestration."""

from __future__ import annotations

import copy
import dataclasses
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import resource
import threading
import time as walltime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulationConfig, config_from_dict
from .core import MatrixSet, SimulationState, TrajectoryBasisFunction
from .dynamics.classical import berry_boris_step, velocity_verlet
from .dynamics.hamiltonian import BerryExactHamiltonian, SaddlePointHamiltonian
from .dynamics.gaussian import gaussian_overlap
from .dynamics.quantum import (
    QuantumPropagationError,
    adaptive_cayley_step,
    metric_norm,
    normalize,
    rk45_step,
)
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
    nac_norm,
    prune_by_overlap,
    projected_coupling,
    trajectory_populations,
)
from .tasks import SerialExecutor, Task, TaskKind, TaskQueue


_WORKER_PROVIDER: ElectronicStructureProvider | None = None


def _initialize_electronic_worker(provider: ElectronicStructureProvider) -> None:
    global _WORKER_PROVIDER
    _WORKER_PROVIDER = provider


def _worker_evaluate(request: ElectronicStructureRequest) -> ElectronicStructureResult:
    if _WORKER_PROVIDER is None:
        raise RuntimeError("electronic worker provider was not initialized")
    return _WORKER_PROVIDER.evaluate(request)


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
                "gradient_mask": _array_payload(electronic.gradient_mask),
                "nacs": _array_payload(electronic.nacs),
                "nac_mask": _array_payload(electronic.nac_mask),
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
                gradient_mask=_array_from_payload(electronic_payload.get("gradient_mask")),
                nacs=_array_from_payload(electronic_payload["nacs"]),
                nac_mask=_array_from_payload(electronic_payload.get("nac_mask")),
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
    classical_energy_references: dict[str, float]
    quantum_energy_reference: float | None
    quantum_norm_reference: float | None = None
    forced_nuclear_time_steps: dict[str, float] = dataclasses.field(default_factory=dict)
    replay_suppressed: dict[tuple[str, int], float] = dataclasses.field(default_factory=dict)
    quantum_energy_gate_active: bool = False
    centroid_nacs: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": [self.key[0], self.key[1]],
            "state": _state_payload(self.state),
            "queue": self.queue,
            "rng_state": self.rng_state,
            "monitor": self.monitor,
            "centroid_wavefunctions": self.centroid_wavefunctions,
            "classical_energy_references": self.classical_energy_references,
            "quantum_energy_reference": self.quantum_energy_reference,
            "quantum_norm_reference": self.quantum_norm_reference,
            "forced_nuclear_time_steps": self.forced_nuclear_time_steps,
            "replay_suppressed": {
                f"{key[0]}:{key[1]}": value
                for key, value in self.replay_suppressed.items()
            },
            "quantum_energy_gate_active": self.quantum_energy_gate_active,
            "centroid_nacs": {
                key: _array_payload(value)
                for key, value in self.centroid_nacs.items()
            },
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
            classical_energy_references={
                str(key): float(value)
                for key, value in payload.get("classical_energy_references", {}).items()
            },
            quantum_energy_reference=(
                None
                if payload.get("quantum_energy_reference") is None
                else float(payload["quantum_energy_reference"])
            ),
            quantum_norm_reference=(
                None
                if payload.get("quantum_norm_reference") is None
                else float(payload["quantum_norm_reference"])
            ),
            forced_nuclear_time_steps={
                str(key): float(value)
                for key, value in payload.get("forced_nuclear_time_steps", {}).items()
            },
            replay_suppressed={
                (key.rsplit(":", 1)[0], int(key.rsplit(":", 1)[1])): float(value)
                for key, value in payload.get("replay_suppressed", {}).items()
            },
            quantum_energy_gate_active=bool(
                payload.get("quantum_energy_gate_active", False)
            ),
            centroid_nacs={
                str(key): _array_from_payload(value)
                for key, value in payload.get("centroid_nacs", {}).items()
            },
        )


@dataclasses.dataclass
class ReplayWindow:
    identifier: str
    frontier_step: int
    frontier_time: float
    entry: SpawnReplaySnapshot
    spawn_task_identifier: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "frontier_step": self.frontier_step,
            "frontier_time": self.frontier_time,
            "entry": self.entry.to_dict(),
            "spawn_task_identifier": self.spawn_task_identifier,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplayWindow":
        return cls(
            identifier=str(payload["identifier"]),
            frontier_step=int(payload["frontier_step"]),
            frontier_time=float(payload["frontier_time"]),
            entry=SpawnReplaySnapshot.from_dict(payload["entry"]),
            spawn_task_identifier=str(payload["spawn_task_identifier"]),
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
        self.centroid_nac_previous: dict[str, np.ndarray] = {}
        self.active_spawn_snapshot: SpawnReplaySnapshot | None = None
        self.active_replay: ReplayWindow | None = None
        self.active_step_snapshot: SpawnReplaySnapshot | None = None
        self._in_replay = False
        self.deferred_spawns: list[SpawnCandidate] = []
        self.replay_suppressed: dict[tuple[str, int], float] = {}
        self.forced_nuclear_time_steps: dict[str, float] = {}
        self.last_quantum_diagnostics: dict[str, Any] = {}
        self.last_matrix_diagnostics: dict[str, Any] = {}
        self.metrics_lock = threading.Lock()
        self.runtime_metrics: dict[str, float] = {
            "electronic_calls": 0.0,
            "electronic_seconds": 0.0,
            "tbf_electronic_calls": 0.0,
            "centroid_electronic_calls": 0.0,
            "scf_seconds": 0.0,
            "casscf_seconds": 0.0,
            "gradient_seconds": 0.0,
            "nac_seconds": 0.0,
            "gradient_evaluations": 0.0,
            "nac_evaluations": 0.0,
            "nac_pairs_screened_by_gap": 0.0,
            "centroid_cache_hits": 0.0,
            "checkpoint_count": 0.0,
            "checkpoint_seconds": 0.0,
            "replay_seconds": 0.0,
        }
        self.classical_energy_references: dict[str, float] = {}
        self.quantum_energy_reference: float | None = None
        self.quantum_norm_reference: float | None = None
        self.quantum_energy_gate_active = False
        requested_workers = max(
            1, int(config.provider_option_dict().get("electronic_workers", 1))
        )
        self.electronic_workers = (
            requested_workers
            if config.provider == "pyscf" and self._coupling_mode() == "nac"
            else 1
        )
        self.run_directory = config.run_directory
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.task_journal = self.run_directory / "task-journal.jsonl"
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
        self.electronic_pool: ProcessPoolExecutor | None = None
        if self.electronic_workers > 1:
            self.electronic_pool = ProcessPoolExecutor(
                max_workers=self.electronic_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_initialize_electronic_worker,
                initargs=(self.provider,),
            )
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

    def close(self) -> None:
        if self.electronic_pool is not None:
            self.electronic_pool.shutdown(wait=True, cancel_futures=True)
            self.electronic_pool = None

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
        gradient_states: tuple[int, ...] | None = None,
        nac_pairs: tuple[tuple[int, int], ...] | None = None,
    ) -> ElectronicStructureRequest:
        properties = {ElectronicProperties.ENERGIES}
        if gradients:
            properties.add(ElectronicProperties.GRADIENTS)
        if self.config.num_states > 1:
            if self._coupling_mode() == "npi":
                properties.add(ElectronicProperties.STATE_OVERLAPS)
            elif nac_pairs is None or len(nac_pairs) > 0:
                properties.add(ElectronicProperties.NACS)
        return ElectronicStructureRequest(
            atoms=self.atoms,
            atomic_numbers=self.atomic_numbers,
            geometry=np.asarray(geometry, dtype=float),
            states=tuple(range(self.config.num_states)),
            properties=frozenset(properties),
            active_state=active_state,
            time=time,
            previous=previous,
            gradient_states=(
                (active_state,) if gradients and gradient_states is None else gradient_states
            ),
            nac_pairs=nac_pairs,
            nac_gap_threshold=self.config.nac_gap_threshold,
        )

    def _evaluate(
        self,
        geometry: np.ndarray,
        active_state: int,
        time: float,
        previous=None,
        *,
        gradients: bool = True,
        gradient_states: tuple[int, ...] | None = None,
        nac_pairs: tuple[tuple[int, int], ...] | None = None,
    ) -> ElectronicStructureResult:
        request = self._request(
            geometry, active_state, time, previous, gradients=gradients,
            gradient_states=gradient_states, nac_pairs=nac_pairs,
        )
        last_error: BaseException | None = None
        for _attempt in range(self.config.electronic_retries + 1):
            try:
                call_started = walltime.perf_counter()
                result = (
                    self.provider.evaluate(request)
                    if self.electronic_pool is None
                    else self.electronic_pool.submit(_worker_evaluate, request).result()
                ).validate(request)
                with self.metrics_lock:
                    timings = result.metadata.get("timings", {})
                    self.runtime_metrics["electronic_calls"] += 1
                    self.runtime_metrics["electronic_seconds"] += (
                        walltime.perf_counter() - call_started
                    )
                    call_kind = (
                        "tbf_electronic_calls"
                        if ElectronicProperties.GRADIENTS in request.properties
                        else "centroid_electronic_calls"
                    )
                    self.runtime_metrics[call_kind] += 1
                    for timing_name in ("scf_seconds", "casscf_seconds"):
                        self.runtime_metrics[timing_name] += float(
                            timings.get(timing_name, 0.0)
                        )
                    for timing_name in ("gradient_seconds", "nac_seconds"):
                        values = timings.get(timing_name, {})
                        self.runtime_metrics[timing_name] += float(
                            sum(values.values()) if isinstance(values, dict) else values
                        )
                    self.runtime_metrics["gradient_evaluations"] += float(
                        np.count_nonzero(result.gradient_mask)
                        if result.gradient_mask is not None else 0
                    )
                    self.runtime_metrics["nac_evaluations"] += float(
                        len(result.metadata.get("evaluated_nac_pairs", ()))
                    )
                    self.runtime_metrics["nac_pairs_screened_by_gap"] += float(
                        len(result.metadata.get("screened_nac_pairs", ()))
                    )
                adopt = getattr(self.provider, "adopt_wavefunction", None)
                if callable(adopt):
                    adopt(result.wavefunction)
                if result.state_overlaps is not None:
                    transform = self.gauge_tracker.align(result.state_overlaps)
                    order = transform.permutation
                    result.energies = result.energies[order]
                    if result.gradients is not None:
                        result.gradients = result.gradients[order]
                        result.gradient_mask = result.gradient_mask[order]
                    if result.nacs is not None:
                        reordered = result.nacs[order][:, order]
                        result.nacs = transform_nacs(reordered, transform.unitary)
                        result.nac_mask = result.nac_mask[order][:, order]
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
            trajectory.positions, trajectory.state, trajectory.time, trajectory.previous_wavefunction,
            gradient_states=(trajectory.state,),
            nac_pairs=tuple(
                (min(trajectory.state, target), max(trajectory.state, target))
                for target in range(self.config.num_states) if target != trajectory.state
            ),
        )
        trajectory.electronic = result
        trajectory.previous_wavefunction = result.wavefunction

    def _trajectory_nac_pairs(self, state: int) -> tuple[tuple[int, int], ...]:
        return tuple(
            (min(state, target), max(state, target))
            for target in range(self.config.num_states) if target != state
        )

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
                nac_pairs=(
                    () if left.state == right.state else
                    ((min(left.state, right.state), max(left.state, right.state)),)
                ),
            )
            self.centroid_cache[key] = result
            if result.wavefunction is not None:
                self.centroid_previous[pair] = result.wavefunction
            if (
                left.state != right.state
                and result.nacs is not None
                and result.nac_mask is not None
                and result.nac_mask[left.state, right.state]
            ):
                state_i, state_j = left.state, right.state
                current_nac = result.nac_between(state_i, state_j).copy()
                previous_nac = self.centroid_nac_previous.get(pair)
                phase = 1.0 + 0.0j
                if previous_nac is not None:
                    overlap = np.vdot(previous_nac.reshape(-1), current_nac.reshape(-1))
                    if abs(overlap) > 1.0e-14:
                        phase = np.exp(-1j * np.angle(overlap))
                        current_nac *= phase
                        result.nacs[state_i, state_j] = current_nac
                        result.nacs[state_j, state_i] = -current_nac.conj()
                self.centroid_nac_previous[pair] = current_nac
                result.metadata["centroid_nac_parallel_transport_phase"] = phase
        else:
            with self.metrics_lock:
                self.runtime_metrics["centroid_cache_hits"] += 1
        return self.centroid_cache[key]

    def _build_matrices(self, dt: float | None = None) -> MatrixSet:
        dt = self.config.time_step if dt is None else float(dt)
        velocities = [trajectory.momenta / trajectory.masses for trajectory in self.state.trajectories]
        forces = [
            -trajectory.electronic.gradient_at_state(trajectory.state)
            for trajectory in self.state.trajectories
        ]
        assembler = (
            BerryExactHamiltonian(self.provider)
            if self.provider.__class__.__name__ == "BerryModel2DParallelTransport"
            else SaddlePointHamiltonian(
                self._coupling_mode(), self.config.pair_overlap_threshold
            )
        )
        total_pairs = len(self.state.trajectories) * (len(self.state.trajectories) - 1) // 2
        active_pairs = 0
        if self.electronic_pool is not None:
            pairs: list[tuple[TrajectoryBasisFunction, TrajectoryBasisFunction, np.ndarray]] = []
            for index, left in enumerate(self.state.trajectories):
                for right in self.state.trajectories[index + 1:]:
                    if abs(gaussian_overlap(left, right, electronic=False)) < self.config.pair_overlap_threshold:
                        continue
                    active_pairs += 1
                    centroid = (
                        left.widths * left.positions + right.widths * right.positions
                    ) / (left.widths + right.widths)
                    if self._centroid_key(left, right, centroid) not in self.centroid_cache:
                        pairs.append((left, right, centroid))
            with ThreadPoolExecutor(max_workers=self.electronic_workers) as threads:
                futures = [
                    threads.submit(self._evaluate_centroid, left, right, centroid)
                    for left, right, centroid in pairs
                ]
                for future in futures:
                    future.result()
        else:
            active_pairs = sum(
                abs(gaussian_overlap(left, right, electronic=False))
                >= self.config.pair_overlap_threshold
                for index, left in enumerate(self.state.trajectories)
                for right in self.state.trajectories[index + 1:]
            )
        matrices = assembler.build(
            self.state.trajectories, self._evaluate_centroid, velocities, forces, dt
        )
        self.state.matrices = matrices
        self.last_matrix_diagnostics = {
            "total_tbf_pairs": total_pairs,
            "active_tbf_pairs": active_pairs,
            "screened_tbf_pairs": total_pairs - active_pairs,
            "pair_overlap_threshold": self.config.pair_overlap_threshold,
        }
        return matrices

    def _coupling_region_active(self) -> bool:
        for trajectory in self.state.trajectories:
            electronic = trajectory.electronic
            if electronic is None or electronic.nacs is None or electronic.nac_mask is None:
                continue
            for target in range(self.config.num_states):
                if target == trajectory.state or not electronic.nac_mask[trajectory.state, target]:
                    continue
                gap = abs(
                    electronic.energies[target]
                    - electronic.energies[trajectory.state]
                )
                if gap > min(self.config.max_energy_gap, self.config.nac_gap_threshold):
                    continue
                nac = electronic.nac_between(trajectory.state, target)
                coupling = (
                    nac_norm(nac)
                    if self.config.spawn_metric == "nac_norm"
                    else projected_coupling(trajectory, nac)
                )
                if coupling >= self.config.spawn_threshold:
                    return True
        return False

    @staticmethod
    def _time_key(value: float) -> str:
        return f"{float(value):.12f}"

    def _nuclear_time_step(self, target_time: float) -> float:
        key = self._time_key(self.state.quantum_time)
        forced = self.forced_nuclear_time_steps.get(key)
        coupling_dt = self.config.coupling_time_step or self.config.time_step
        selected = forced or (
            coupling_dt if self._coupling_region_active() else self.config.time_step
        )
        return min(float(selected), target_time - self.state.quantum_time)

    def _refined_nuclear_time_step(self, current: float) -> float | None:
        coupling_dt = self.config.coupling_time_step or self.config.time_step
        floor = self.config.minimum_nuclear_time_step or coupling_dt
        candidate = max(0.5 * current, floor)
        return None if candidate >= current - 1.0e-15 else candidate

    @staticmethod
    def _is_refinable_electronic_failure(exc: BaseException) -> bool:
        return any(
            marker in str(exc)
            for marker in (
                "energy/gradient continuity failure",
                "CI-root continuity failure",
            )
        )

    def _regularization_ladder(self) -> tuple[float, ...]:
        """Return deterministic overlap cutoffs, starting with the configured value."""
        configured = float(self.config.regularization_threshold)
        canonical = (1.0e-4, 1.0e-5, 1.0e-6, 1.0e-8)
        if np.isclose(configured, canonical[0], rtol=0.0, atol=1.0e-16):
            return canonical
        return (configured,)

    def _retry_nuclear_step(
        self,
        snapshot: SpawnReplaySnapshot | None,
        current_dt: float,
        reason: str,
    ) -> bool:
        refined = self._refined_nuclear_time_step(current_dt)
        if snapshot is None or refined is None:
            return False
        self._restore_spawn_snapshot(snapshot)
        self.forced_nuclear_time_steps[self._time_key(self.state.quantum_time)] = refined
        self.state.events.append({
            "kind": "nuclear_step_refinement",
            "time": self.state.quantum_time,
            "from_dt": current_dt,
            "to_dt": refined,
            "reason": reason,
        })
        self.active_step_snapshot = None
        self._checkpoint()
        return True

    @staticmethod
    def _trajectory_energy(trajectory: TrajectoryBasisFunction) -> float:
        if trajectory.electronic is None:
            raise RuntimeError(f"trajectory {trajectory.label} has no electronic energy")
        return float(
            np.sum(trajectory.momenta**2 / (2.0 * trajectory.masses))
            + trajectory.electronic.energies[trajectory.state]
        )

    def _check_classical_energies(self) -> None:
        tolerance = (
            self.config.energy_tolerance
            if self.config.classical_energy_tolerance is None
            else self.config.classical_energy_tolerance
        )
        for trajectory in self.state.trajectories:
            energy = self._trajectory_energy(trajectory)
            reference = self.classical_energy_references.setdefault(
                trajectory.identifier, energy
            )
            comparison_limit = (
                tolerance + self.config.classical_energy_numerical_margin
            )
            if abs(energy - reference) > comparison_limit:
                raise RuntimeError(
                    "classical energy violation: "
                    f"trajectory={trajectory.label}, reference={reference}, "
                    f"current={energy}, drift={energy - reference}, "
                    f"tolerance={tolerance}, "
                    f"numerical_margin={self.config.classical_energy_numerical_margin}"
                )

    def _check_quantum_energy(self) -> float:
        if self.state.matrices is None:
            raise RuntimeError("quantum energy requires assembled matrices")
        denominator = np.vdot(
            self.state.amplitudes,
            self.state.matrices.overlap @ self.state.amplitudes,
        )
        energy = float(np.real(
            np.vdot(
                self.state.amplitudes,
                self.state.matrices.hamiltonian @ self.state.amplitudes,
            ) / denominator
        ))
        if self.quantum_energy_reference is None:
            self.quantum_energy_reference = energy
        else:
            drift = energy - self.quantum_energy_reference
            exceeded = abs(drift) > self.config.energy_tolerance
            if exceeded and self.config.quantum_energy_policy == "error":
                raise RuntimeError(
                    "quantum energy violation: "
                    f"reference={self.quantum_energy_reference}, current={energy}, "
                    f"drift={drift}"
                )
            if exceeded and not self.quantum_energy_gate_active:
                self.state.events.append({
                    "kind": "quantum_energy_threshold",
                    "time": self.state.quantum_time,
                    "reference": self.quantum_energy_reference,
                    "current": energy,
                    "drift": drift,
                    "tolerance": self.config.energy_tolerance,
                    "policy": self.config.quantum_energy_policy,
                })
            self.quantum_energy_gate_active = exceeded
        return energy

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
                    gradient_states=(trajectory.state,),
                    nac_pairs=self._trajectory_nac_pairs(trajectory.state),
                )
                return endpoint.gradient_at_state(trajectory.state)

            trajectory.positions, trajectory.momenta, _ = velocity_verlet(
                trajectory.positions, trajectory.momenta, trajectory.masses, dt, gradient0, gradient_at
            )
        trajectory.time += dt
        trajectory.electronic = endpoint
        if endpoint is not None:
            trajectory.previous_wavefunction = endpoint.wavefunction
        self.state.matrices = None

    def _current_snapshot(self, key: tuple[str, int]) -> SpawnReplaySnapshot:
        return SpawnReplaySnapshot(
            key=key,
            state=_state_from_payload(_state_payload(self.state)),
            queue=copy.deepcopy(self.queue.to_dict()),
            rng_state=copy.deepcopy(self.rng.bit_generator.state),
            monitor=copy.deepcopy(self.monitor.to_dict()),
            centroid_wavefunctions={
                name: wavefunction.identifier
                for name, wavefunction in self.centroid_previous.items()
            },
            classical_energy_references=copy.deepcopy(
                self.classical_energy_references
            ),
            quantum_energy_reference=self.quantum_energy_reference,
            quantum_norm_reference=self.quantum_norm_reference,
            forced_nuclear_time_steps=copy.deepcopy(self.forced_nuclear_time_steps),
            replay_suppressed=copy.deepcopy(self.replay_suppressed),
            quantum_energy_gate_active=self.quantum_energy_gate_active,
            centroid_nacs={
                key: value.copy()
                for key, value in self.centroid_nac_previous.items()
            },
        )

    def _capture_spawn_snapshot(self, key: tuple[str, int]) -> None:
        self.active_spawn_snapshot = self._current_snapshot(key)

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
        self.classical_energy_references = copy.deepcopy(
            snapshot.classical_energy_references
        )
        self.quantum_energy_reference = snapshot.quantum_energy_reference
        self.quantum_norm_reference = snapshot.quantum_norm_reference
        self.forced_nuclear_time_steps = copy.deepcopy(
            snapshot.forced_nuclear_time_steps
        )
        self.replay_suppressed = copy.deepcopy(snapshot.replay_suppressed)
        self.quantum_energy_gate_active = snapshot.quantum_energy_gate_active
        self.centroid_nac_previous = {
            key: value.copy() for key, value in snapshot.centroid_nacs.items()
        }

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
                if (
                    trajectory.electronic.nac_mask is None
                    or not trajectory.electronic.nac_mask[trajectory.state, target]
                ):
                    continue
                nac = trajectory.electronic.nac_between(trajectory.state, target)
                coupling = (
                    nac_norm(nac)
                    if self.config.spawn_metric == "nac_norm"
                    else projected_coupling(trajectory, nac)
                )
                candidate = SpawnCandidate(
                    index, target, coupling, trajectory.time,
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

    def _backpropagate_spawn_child(
        self,
        child: TrajectoryBasisFunction,
        entry_time: float,
    ) -> TrajectoryBasisFunction:
        """Backpropagate a provisional child with local transactional refinement.

        Spawn maxima and threshold entries can straddle the same sharp electronic
        regions that require refined ordinary nuclear steps.  A rejected trial
        must not mutate the accepted child history, so every substep is attempted
        on a private copy and committed only after its endpoint passes the
        provider's continuity checks.
        """
        if child.electronic is None:
            child.electronic = self._evaluate(
                child.positions,
                child.state,
                child.time,
                child.previous_wavefunction,
                gradient_states=(child.state,),
                nac_pairs=self._trajectory_nac_pairs(child.state),
            )
            child.previous_wavefunction = child.electronic.wavefunction

        coupling_dt = self.config.coupling_time_step or self.config.time_step
        while child.time - 1.0e-12 > entry_time:
            trial_dt = min(coupling_dt, child.time - entry_time)
            while True:
                trial_child = copy.deepcopy(child)
                self.state.trajectories.append(trial_child)
                try:
                    self._propagate_trajectory(
                        len(self.state.trajectories) - 1, -trial_dt
                    )
                except ElectronicStructureError as exc:
                    refinable = self._is_refinable_electronic_failure(exc)
                    refined = (
                        self._refined_nuclear_time_step(trial_dt)
                        if refinable
                        else None
                    )
                    if refined is None:
                        raise
                    self.state.events.append({
                        "kind": "spawn_backprop_refinement",
                        "time": child.time,
                        "from_dt": trial_dt,
                        "to_dt": refined,
                        "threshold_entry_time": entry_time,
                        "child": child.label,
                        "reason": str(exc),
                    })
                    trial_dt = refined
                    continue
                finally:
                    self.state.trajectories.pop()
                child = trial_child
                break
        return child

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
            self.state.events.append({
                "kind": "failed_spawn", "time": candidate.time,
                "reason": "maximum_overlap",
            })
            self.active_spawn_snapshot = None
            return
        parent_energy = float(
            np.sum(candidate.momenta**2 / (2.0 * parent.masses))
            + candidate.energies[parent.state]
        )
        child_energy = float(
            np.sum(child.momenta**2 / (2.0 * child.masses))
            + candidate.energies[child.state]
        )
        energy_mismatch = abs(child_energy - parent_energy)
        if energy_mismatch > 1.0e-8:
            raise RuntimeError(
                f"spawn energy mismatch {energy_mismatch} Eh exceeds 1e-8 Eh"
            )
        # Build the child's missing history back to the threshold-entry time.
        entry_time = candidate.entry_time if candidate.entry_time is not None else candidate.time
        backprop_event_start = len(self.state.events)
        child = self._backpropagate_spawn_child(child, entry_time)
        backprop_events = copy.deepcopy(self.state.events[backprop_event_start:])

        snapshot = self.active_spawn_snapshot
        if snapshot is None:
            raise RuntimeError("spawn replay snapshot is missing")
        if overlaps_existing(child, snapshot.state.trajectories, self.config.spawn_overlap_max):
            self.state.events.append({
                "kind": "failed_spawn", "time": candidate.time,
                "reason": "entry_overlap", "threshold_entry_time": entry_time,
                "parent": parent.label, "target_state": child.state,
            })
            self.active_spawn_snapshot = None
            return
        frontier_time = self.state.quantum_time
        frontier_step = self.state.step
        replay_key = snapshot.key
        self._restore_spawn_snapshot(snapshot)
        self.active_spawn_snapshot = None
        self.state.events.extend(backprop_events)
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
            "energy_mismatch": energy_mismatch,
            "child_amplitude_at_insertion": 0.0,
        })
        self.classical_energy_references[child.identifier] = self._trajectory_energy(child)
        self.state.matrices = None
        self._build_matrices()
        window_id = (
            f"{self.state.step:08d}-{replay_key[0][:12]}-"
            f"s{replay_key[1]}-to-{frontier_step:08d}"
        )
        entry_snapshot = self._current_snapshot(replay_key)
        self.active_replay = ReplayWindow(
            identifier=window_id,
            frontier_step=frontier_step,
            frontier_time=frontier_time,
            entry=entry_snapshot,
            spawn_task_identifier=(
                f"spawn:{candidate.parent_index}:{candidate.target_state}:{candidate.time}"
            ),
        )
        self.writer.begin_replay(
            window_id, entry_step=self.state.step, frontier_step=frontier_step
        )
        self.writer.write_step(
            self.state, self.config.num_states, {"spawn_insertion": 1}
        )
        self._checkpoint()
        self._run_active_replay(reset_staging=False)

    def _run_active_replay(self, *, reset_staging: bool) -> None:
        window = self.active_replay
        if window is None:
            return
        if reset_staging:
            self._restore_spawn_snapshot(window.entry)
            self.writer.begin_replay(
                window.identifier,
                entry_step=self.state.step,
                frontier_step=window.frontier_step,
            )
            self.writer.reset_replay(window.identifier)
            self.writer.write_step(
                self.state, self.config.num_states, {"spawn_insertion": 1}
            )
        self._in_replay = True
        replay_started = walltime.perf_counter()
        try:
            self.propagate(stop_after_time=window.frontier_time)
        except BaseException:
            self._restore_spawn_snapshot(window.entry)
            self.writer.reset_replay(window.identifier)
            self.writer.write_step(
                self.state, self.config.num_states, {"spawn_insertion": 1}
            )
            self._checkpoint()
            raise
        finally:
            self._in_replay = False
            with self.metrics_lock:
                self.runtime_metrics["replay_seconds"] += walltime.perf_counter() - replay_started
        self.writer.commit_replay(window.identifier, self.state.step)
        self.active_replay = None
        if reset_staging and window.spawn_task_identifier not in self.queue.completed:
            self.queue.completed.add(window.spawn_task_identifier)
            self.state.events.append({
                "kind": "task", "task": TaskKind.SPAWN.value,
                "id": window.spawn_task_identifier,
                "time": self.state.quantum_time,
            })
        self._checkpoint()

        deferred, self.deferred_spawns = self.deferred_spawns, []
        for candidate in deferred:
            task = self.queue.add(
                TaskKind.SPAWN,
                candidate.time,
                f"spawn:{candidate.parent_index}:{candidate.target_state}:{candidate.time}",
            )
            self._execute(task, lambda current, item=candidate: self._spawn(item))

    def _checkpoint(self) -> None:
        checkpoint_started = walltime.perf_counter()
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
        if self.active_replay is not None:
            references.update(
                trajectory.previous_wavefunction.identifier
                for trajectory in self.active_replay.entry.state.trajectories
                if trajectory.previous_wavefunction is not None
            )
            references.update(self.active_replay.entry.centroid_wavefunctions.values())
        if self.active_step_snapshot is not None:
            references.update(
                trajectory.previous_wavefunction.identifier
                for trajectory in self.active_step_snapshot.state.trajectories
                if trajectory.previous_wavefunction is not None
            )
            references.update(self.active_step_snapshot.centroid_wavefunctions.values())
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
                "centroid_nacs": {
                    key: _array_payload(value)
                    for key, value in self.centroid_nac_previous.items()
                },
                "active_spawn_snapshot": (
                    self.active_spawn_snapshot.to_dict()
                    if self.active_spawn_snapshot is not None else None
                ),
                "active_replay": (
                    self.active_replay.to_dict()
                    if self.active_replay is not None else None
                ),
                "active_step_snapshot": (
                    self.active_step_snapshot.to_dict()
                    if self.active_step_snapshot is not None else None
                ),
                "last_quantum_diagnostics": copy.deepcopy(
                    self.last_quantum_diagnostics
                ),
                "classical_energy_references": self.classical_energy_references,
                "quantum_energy_reference": self.quantum_energy_reference,
                "quantum_norm_reference": self.quantum_norm_reference,
                "quantum_energy_gate_active": self.quantum_energy_gate_active,
                "replay_suppressed": [
                    [key[0], key[1], frontier]
                    for key, frontier in self.replay_suppressed.items()
                ],
                "forced_nuclear_time_steps": self.forced_nuclear_time_steps,
                "runtime_metrics": self.runtime_metrics,
            },
        )
        with self.metrics_lock:
            self.runtime_metrics["checkpoint_count"] += 1
            self.runtime_metrics["checkpoint_seconds"] += (
                walltime.perf_counter() - checkpoint_started
            )

    def _execute(self, task: Task, action) -> None:
        try:
            if not task.dependencies <= self.queue.completed:
                missing = sorted(task.dependencies - self.queue.completed)
                raise RuntimeError(f"task {task.identifier} has incomplete dependencies: {missing}")
            self.executor.execute(task, action)
            self.queue.complete(task)
            self.state.events.append({"kind": "task", "task": task.kind.value, "id": task.identifier, "time": self.state.quantum_time})
            self._journal_task(task, "completed")
        except BaseException as exc:
            # Replay owns its queue transaction.  Its exception handler restores
            # the exact threshold-entry queue, so do not contaminate that queue
            # with provisional task failures.
            if self.active_replay is None:
                self.queue.fail(task, exc)
            self._journal_task(task, "failed", str(exc))
            self._checkpoint()
            raise

    def _journal_task(self, task: Task, status: str, error: str | None = None) -> None:
        record = {
            "id": task.identifier,
            "kind": task.kind.value,
            "scheduled_time": task.priority[0],
            "simulation_time": self.state.quantum_time,
            "status": status,
        }
        if error is not None:
            record["error"] = error
        with self.task_journal.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _execute_trajectory_batch(self, tasks: list[Task], dt: float) -> None:
        if self.electronic_pool is None or len(tasks) < 2:
            for index, task in enumerate(tasks):
                self._execute(
                    task, lambda current, idx=index: self._propagate_trajectory(idx, dt)
                )
            return
        snapshot = self._current_snapshot(("trajectory-batch", -1))
        try:
            with ThreadPoolExecutor(max_workers=self.electronic_workers) as threads:
                futures = [
                    threads.submit(self._propagate_trajectory, index, dt)
                    for index in range(len(tasks))
                ]
                for future in futures:
                    future.result()
        except BaseException as exc:
            self._restore_spawn_snapshot(snapshot)
            self.state.events.append({
                "kind": "task_failure",
                "task": "trajectory_batch",
                "time": self.state.quantum_time,
                "error": str(exc),
            })
            self._checkpoint()
            raise
        for task in tasks:
            self.queue.complete(task)
            self.state.events.append({
                "kind": "task", "task": task.kind.value,
                "id": task.identifier, "time": self.state.quantum_time,
            })
            self._journal_task(task, "completed")

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
        self.centroid_nac_previous = {
            str(key): _array_from_payload(value)
            for key, value in metadata.get("runtime", {}).get(
                "centroid_nacs", {}
            ).items()
        }
        active_spawn = metadata.get("runtime", {}).get("active_spawn_snapshot")
        self.active_spawn_snapshot = (
            SpawnReplaySnapshot.from_dict(active_spawn) if active_spawn is not None else None
        )
        active_replay = metadata.get("runtime", {}).get("active_replay")
        self.active_replay = (
            ReplayWindow.from_dict(active_replay) if active_replay is not None else None
        )
        active_step = metadata.get("runtime", {}).get("active_step_snapshot")
        self.active_step_snapshot = (
            SpawnReplaySnapshot.from_dict(active_step) if active_step is not None else None
        )
        self.last_quantum_diagnostics = dict(
            metadata.get("runtime", {}).get("last_quantum_diagnostics", {})
        )
        self.classical_energy_references = {
            str(key): float(value)
            for key, value in metadata.get("runtime", {}).get(
                "classical_energy_references", {}
            ).items()
        }
        quantum_reference = metadata.get("runtime", {}).get("quantum_energy_reference")
        self.quantum_energy_reference = (
            None if quantum_reference is None else float(quantum_reference)
        )
        norm_reference = metadata.get("runtime", {}).get("quantum_norm_reference")
        self.quantum_norm_reference = (
            None if norm_reference is None else float(norm_reference)
        )
        self.quantum_energy_gate_active = bool(
            metadata.get("runtime", {}).get("quantum_energy_gate_active", False)
        )
        self.replay_suppressed = {
            (str(parent), int(target)): float(frontier)
            for parent, target, frontier in metadata.get("runtime", {}).get(
                "replay_suppressed", []
            )
        }
        self.forced_nuclear_time_steps = {
            str(key): float(value)
            for key, value in metadata.get("runtime", {}).get(
                "forced_nuclear_time_steps", {}
            ).items()
        }
        self.runtime_metrics.update({
            str(key): float(value)
            for key, value in metadata.get("runtime", {}).get(
                "runtime_metrics", {}
            ).items()
        })
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
                    gradient_mask=arrays.get(f"t{index}_gradient_mask"),
                    nac_mask=arrays.get(f"t{index}_nac_mask"),
                )
            trajectories.append(trajectory)
        matrices = None
        if all(key in arrays for key in ("H", "S", "Sdot")):
            matrices = MatrixSet(arrays["S"], arrays["H"], arrays["Sdot"])
        return SimulationState(
            trajectories, arrays["amplitudes"], metadata["quantum_time"], metadata["step"], matrices,
            metadata.get("events", []),
        )

    def propagate(
        self,
        *,
        stop_after_step: int | None = None,
        stop_after_time: float | None = None,
    ) -> SimulationResult:
        if self.active_replay is not None and not self._in_replay:
            self._run_active_replay(reset_staging=True)
        elif self.active_step_snapshot is not None and not self._in_replay:
            self._restore_spawn_snapshot(self.active_step_snapshot)
            self.active_step_snapshot = None
        if not self.is_restart and self.state.step == 0:
            for index in range(len(self.state.trajectories)):
                self._evaluate_trajectory(index)
            self._build_matrices()
            self._check_classical_energies()
            self._check_quantum_energy()
            self.writer.write_step(self.state, self.config.num_states)
            self._checkpoint()
        target_time = self.config.simulation_time
        if stop_after_time is not None:
            target_time = min(target_time, max(self.state.quantum_time, float(stop_after_time)))
        target_step = None if stop_after_step is None else max(self.state.step, int(stop_after_step))
        while (
            self.state.quantum_time < target_time - 1.0e-12
            and (target_step is None or self.state.step < target_step)
        ):
            self.centroid_cache.clear()
            for index, trajectory in enumerate(self.state.trajectories):
                if trajectory.electronic is None:
                    self._evaluate_trajectory(index)

            dt = self._nuclear_time_step(target_time)
            start_in_coupling = self._coupling_region_active()
            start_matrices = self.state.matrices or self._build_matrices(dt)
            removed = prune_by_overlap(self.state, self.config.overlap_threshold)
            if removed:
                self.state.events.append({
                    "kind": "prune", "time": self.state.quantum_time, "indices": removed,
                })
                start_matrices = self._build_matrices(dt)

            for candidate in self._observe_spawning():
                if self._in_replay:
                    self.deferred_spawns.append(candidate)
                    continue
                task = self.queue.add(TaskKind.SPAWN, candidate.time, f"spawn:{candidate.parent_index}:{candidate.target_state}:{candidate.time}")
                self._execute(task, lambda current, item=candidate: self._spawn(item))
                start_matrices = self._build_matrices(dt)

            self.active_step_snapshot = self._current_snapshot(
                (f"outer-step-{self.state.step}", -1)
            )
            if not self._in_replay:
                self._checkpoint()

            propagation_tasks = []
            for trajectory in list(self.state.trajectories):
                task = self.queue.add(
                    TaskKind.TRAJECTORY,
                    trajectory.time,
                    f"prop:{self.state.step}:{trajectory.identifier}",
                )
                propagation_tasks.append(task)
            try:
                self._execute_trajectory_batch(propagation_tasks, dt)
            except ElectronicStructureError as exc:
                # A finite-step energy/gradient consistency rejection may be
                # curvature or ambiguous same-root phase tracking. Retry the
                # entire interval transactionally on the same adiabatic branch.
                # Active-space and other electronic continuity failures remain
                # hard stops.
                if (
                    self._is_refinable_electronic_failure(exc)
                    and self._retry_nuclear_step(
                        self.active_step_snapshot, dt, str(exc)
                    )
                ):
                    continue
                raise
            self.state.quantum_time += dt
            self.state.step += 1
            self.centroid_cache.clear()
            for index, trajectory in enumerate(self.state.trajectories):
                if trajectory.electronic is None:
                    self._evaluate_trajectory(index)
            coupling_dt = self.config.coupling_time_step or self.config.time_step
            if (
                dt > coupling_dt + 1.0e-15
                and not start_in_coupling
                and self._coupling_region_active()
            ):
                if self._retry_nuclear_step(
                    self.active_step_snapshot, dt, "coupling_region_entry"
                ):
                    continue
            try:
                end_matrices = self._build_matrices(dt)
            except ElectronicStructureError as exc:
                # Interstate centroid histories are independent electronic
                # paths.  A root-continuity rejection there must refine the
                # same complete nuclear interval as a rejected TBF endpoint.
                if (
                    self._is_refinable_electronic_failure(exc)
                    and self._retry_nuclear_step(
                        self.active_step_snapshot, dt, str(exc)
                    )
                ):
                    continue
                raise
            # Local force-quadrature and electronic-continuity failures drive
            # transactional step refinement.  The global classical-energy gate
            # remains a hard failsafe; using it to switch Verlet maps causes
            # shadow-Hamiltonian boundary hugging.
            self._check_classical_energies()

            quantum = self.queue.add(
                TaskKind.QUANTUM,
                self.state.quantum_time,
                f"quantum:{self.state.step - 1}",
                dependencies=[task.identifier for task in propagation_tasks],
            )

            def quantum_action(_task: Task) -> None:
                before = metric_norm(self.state.amplitudes, start_matrices.overlap)
                if self.quantum_norm_reference is None:
                    self.quantum_norm_reference = before
                if self.config.quantum_integrator == "cayley":
                    rejected_thresholds = []
                    failure: QuantumPropagationError | None = None
                    for regularization in self._regularization_ladder():
                        try:
                            result = adaptive_cayley_step(
                                self.state.amplitudes,
                                start_matrices,
                                end_matrices,
                                dt,
                                threshold=regularization,
                                convergence_tolerance=self.config.norm_tolerance,
                                norm_tolerance=min(self.config.norm_tolerance, 1.0e-10),
                                min_time_step=self.config.min_time_step,
                            )
                            break
                        except QuantumPropagationError as exc:
                            failure = exc
                            rejected_thresholds.append(regularization)
                    else:
                        assert failure is not None
                        result = failure.result
                        self.last_quantum_diagnostics = {
                            "quantum_substeps": result.substeps,
                            "raw_norm_before": result.norm_before,
                            "raw_norm_after": result.norm_after_raw,
                            "quantum_convergence_error": result.convergence_error,
                            "metric_compatibility_residual": result.metric_residual,
                            "retained_overlap_eigenvalues": result.retained_overlap_eigenvalues,
                            "quantum_converged": 0,
                            "regularization_threshold": rejected_thresholds[-1],
                            "rejected_regularization_thresholds": rejected_thresholds,
                        }
                        raise failure
                    self.state.amplitudes = result.amplitudes
                    self.last_quantum_diagnostics = {
                        "quantum_substeps": result.substeps,
                        "raw_norm_before": result.norm_before,
                        "raw_norm_after": result.norm_after_raw,
                        "quantum_convergence_error": result.convergence_error,
                        "metric_compatibility_residual": result.metric_residual,
                        "retained_overlap_eigenvalues": result.retained_overlap_eigenvalues,
                        "quantum_converged": 1,
                        "regularization_threshold": regularization,
                        "rejected_regularization_thresholds": rejected_thresholds,
                    }
                else:
                    propagated = rk45_step(
                        self.state.amplitudes,
                        start_matrices,
                        dt,
                        self.config.regularization_threshold,
                        end_matrices,
                    )
                    after = metric_norm(propagated, end_matrices.overlap)
                    if abs(after - before) > min(self.config.norm_tolerance, 1.0e-10):
                        raise RuntimeError(
                            f"quantum norm violation: before={before}, after={after}"
                        )
                    self.state.amplitudes = normalize(propagated, end_matrices.overlap)
                    self.last_quantum_diagnostics = {
                        "quantum_substeps": 1,
                        "raw_norm_before": before,
                        "raw_norm_after": after,
                        "quantum_convergence_error": 0.0,
                        "metric_compatibility_residual": np.nan,
                        "quantum_converged": 1,
                    }
                self.state.matrices = end_matrices
                population_sum = float(
                    np.sum(self.state.populations(num_states=self.config.num_states))
                )
                endpoint_norm = metric_norm(
                    self.state.amplitudes, end_matrices.overlap
                )
                absolute_norm_tolerance = min(self.config.norm_tolerance, 1.0e-10)
                if abs(endpoint_norm - self.quantum_norm_reference) > absolute_norm_tolerance:
                    raise RuntimeError(
                        "cumulative quantum norm violation: "
                        f"reference={self.quantum_norm_reference}, "
                        f"current={endpoint_norm}, tolerance={absolute_norm_tolerance}"
                    )
                if abs(population_sum - endpoint_norm) > 1.0e-10:
                    raise RuntimeError(
                        "state population sum is inconsistent with metric norm: "
                        f"populations={population_sum}, norm={endpoint_norm}"
                    )
                quantum_energy = self._check_quantum_energy()
                self.last_quantum_diagnostics["quantum_energy_checked"] = quantum_energy
                self.last_quantum_diagnostics["state_population_sum"] = population_sum

            try:
                self._execute(quantum, quantum_action)
            except RuntimeError as exc:
                if "energy violation" in str(exc) and self._retry_nuclear_step(
                    self.active_step_snapshot, dt, str(exc)
                ):
                    continue
                raise
            self.last_quantum_diagnostics.update(self.last_matrix_diagnostics)
            with self.metrics_lock:
                self.last_quantum_diagnostics.update(self.runtime_metrics)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            self.last_quantum_diagnostics["peak_rss_bytes"] = float(usage.ru_maxrss)
            process_times = os.times()
            self.last_quantum_diagnostics["cpu_seconds"] = float(
                process_times.user + process_times.system
                + process_times.children_user + process_times.children_system
            )
            self.last_quantum_diagnostics["nuclear_time_step"] = dt
            if (
                self.state.step % self.config.output_every == 0
                or self.state.quantum_time >= self.config.simulation_time - 1.0e-12
            ):
                output = self.queue.add(
                    TaskKind.OUTPUT,
                    self.state.quantum_time,
                    f"output:{self.state.step}",
                    dependencies=[quantum.identifier],
                )
                self._execute(output, lambda current: self.writer.write_step(
                    self.state,
                    self.config.num_states,
                    self.last_quantum_diagnostics,
                ))
            if not self._in_replay:
                self.active_step_snapshot = None
                self._checkpoint()
        if self.state.quantum_time < self.config.simulation_time - 1.0e-12:
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
            self.writer.write_step(
                self.state, self.config.num_states, self.last_quantum_diagnostics
            )
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
    runner = SimulationRunner(config, provider, restart=restart)
    try:
        return runner.propagate()
    finally:
        runner.close()


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
