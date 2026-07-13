"""Core data structures for trajectory-basis dynamics."""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

import numpy as np

from .electronic.base import ElectronicStructureResult, WavefunctionState


@dataclasses.dataclass
class TrajectoryBasisFunction:
    positions: np.ndarray
    momenta: np.ndarray
    widths: np.ndarray
    masses: np.ndarray
    state: int
    time: float = 0.0
    label: str = "00"
    identifier: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    generation: int = 0
    phase: float = 0.0
    previous_wavefunction: WavefunctionState | None = None
    electronic: ElectronicStructureResult | None = None
    last_spawn_time: float = -np.inf
    spawn_count: int = 0

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.momenta = np.asarray(self.momenta, dtype=float)
        if self.positions.shape != self.momenta.shape or self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions and momenta must both have shape (natom, 3)")
        self.widths = np.asarray(self.widths, dtype=float).reshape(self.positions.shape)
        self.masses = np.asarray(self.masses, dtype=float).reshape(self.positions.shape)
        if np.any(self.widths <= 0) or np.any(self.masses <= 0):
            raise ValueError("widths and masses must be positive")
        if not np.all(np.isfinite(self.positions)) or not np.all(np.isfinite(self.momenta)):
            raise ValueError("trajectory positions and momenta must be finite")

    @property
    def ndof(self) -> int:
        return self.positions.size

    def copy_child(self, state: int, momenta: np.ndarray) -> "TrajectoryBasisFunction":
        label = f"{self.label}b{self.spawn_count + 1}"
        identifier = uuid.uuid5(
            uuid.NAMESPACE_URL, f"aims-berry:{self.identifier}:{self.spawn_count + 1}:{state}"
        ).hex
        return TrajectoryBasisFunction(
            positions=self.positions.copy(),
            momenta=np.asarray(momenta, dtype=float).copy(),
            widths=self.widths.copy(),
            masses=self.masses.copy(),
            state=state,
            time=self.time,
            label=label,
            identifier=identifier,
            parent_id=self.identifier,
            generation=self.generation + 1,
        )


@dataclasses.dataclass
class MatrixSet:
    overlap: np.ndarray
    hamiltonian: np.ndarray
    sdot: np.ndarray

    def __post_init__(self) -> None:
        self.overlap = np.asarray(self.overlap, dtype=np.complex128)
        self.hamiltonian = np.asarray(self.hamiltonian, dtype=np.complex128)
        self.sdot = np.asarray(self.sdot, dtype=np.complex128)
        if self.overlap.ndim != 2 or self.overlap.shape[0] != self.overlap.shape[1]:
            raise ValueError("matrix set must be square")
        if self.hamiltonian.shape != self.overlap.shape or self.sdot.shape != self.overlap.shape:
            raise ValueError("H, S, and Sdot must have identical shapes")
        if not all(np.all(np.isfinite(value)) for value in (self.overlap, self.hamiltonian, self.sdot)):
            raise ValueError("matrix set contains non-finite values")
        if np.linalg.norm(self.overlap - self.overlap.conj().T) > 1.0e-10:
            raise ValueError("overlap matrix is not Hermitian")
        if np.linalg.norm(self.hamiltonian - self.hamiltonian.conj().T) > 1.0e-10:
            raise ValueError("Hamiltonian is not Hermitian")

    @property
    def effective(self) -> np.ndarray:
        return self.hamiltonian - 1j * self.sdot


@dataclasses.dataclass
class SimulationState:
    trajectories: list[TrajectoryBasisFunction]
    amplitudes: np.ndarray
    quantum_time: float = 0.0
    step: int = 0
    matrices: MatrixSet | None = None
    events: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
        if self.amplitudes.shape != (len(self.trajectories),):
            raise ValueError("one amplitude is required per trajectory")
        if not np.all(np.isfinite(self.amplitudes)):
            raise ValueError("amplitudes must be finite")

    def add_trajectory(self, trajectory: TrajectoryBasisFunction, amplitude: complex = 0j) -> int:
        self.trajectories.append(trajectory)
        self.amplitudes = np.append(self.amplitudes, complex(amplitude))
        return len(self.trajectories) - 1

    def populations(self, overlap: np.ndarray | None = None, num_states: int | None = None) -> np.ndarray:
        overlap = overlap if overlap is not None else (self.matrices.overlap if self.matrices else np.eye(len(self.amplitudes)))
        nstates = num_states or (max(t.state for t in self.trajectories) + 1)
        pops = np.zeros(nstates, dtype=float)
        for state in range(nstates):
            idx = [i for i, t in enumerate(self.trajectories) if t.state == state]
            if idx:
                c = self.amplitudes[idx]
                pops[state] = float(np.real(np.vdot(c, overlap[np.ix_(idx, idx)] @ c)))
        return pops


@dataclasses.dataclass
class GaussianBasis:
    """Validated collection of trajectory basis functions."""

    trajectories: list[TrajectoryBasisFunction]

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("a Gaussian basis cannot be empty")
        shape = self.trajectories[0].positions.shape
        if any(trajectory.positions.shape != shape for trajectory in self.trajectories):
            raise ValueError("all Gaussian basis functions must use the same nuclear dimensions")

    def overlap(self) -> np.ndarray:
        from .dynamics.gaussian import overlap_matrix
        return overlap_matrix(self.trajectories)
