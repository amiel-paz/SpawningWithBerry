"""Spawning criteria, energy-shell momentum adjustment, and pruning."""

from __future__ import annotations

import dataclasses

import numpy as np

from .core import SimulationState, TrajectoryBasisFunction
from .dynamics.gaussian import gaussian_overlap


@dataclasses.dataclass(frozen=True)
class SpawnCandidate:
    parent_index: int
    target_state: int
    coupling: float
    time: float
    positions: np.ndarray
    momenta: np.ndarray
    nac: np.ndarray
    energies: np.ndarray
    entry_time: float | None = None
    entry_positions: np.ndarray | None = None
    entry_momenta: np.ndarray | None = None
    parent_id: str | None = None


def trajectory_populations(state: SimulationState) -> np.ndarray:
    overlap = state.matrices.overlap if state.matrices is not None else np.eye(len(state.trajectories))
    return np.real(np.conjugate(state.amplitudes) * (overlap @ state.amplitudes))


def projected_coupling(trajectory: TrajectoryBasisFunction, nac: np.ndarray) -> float:
    velocity = trajectory.momenta / trajectory.masses
    phase = np.exp(-1j * np.angle(np.vdot(nac.reshape(-1), velocity.reshape(-1)) or 1.0))
    return float(abs(np.sum((phase * nac).real * velocity)))


def nac_norm(nac: np.ndarray) -> float:
    """Euclidean NAC magnitude used by the canonical molecular AIMS protocol."""
    return float(np.linalg.norm(np.asarray(nac)))


def energy_matched_momentum(
    momentum: np.ndarray,
    masses: np.ndarray,
    direction: np.ndarray,
    parent_energy: float,
    target_energy: float,
) -> np.ndarray | None:
    direction = np.asarray(direction, dtype=np.complex128)
    phase_ref = np.vdot(direction.reshape(-1), momentum.reshape(-1) / masses.reshape(-1))
    direction = (np.exp(-1j * np.angle(phase_ref or 1.0)) * direction).real
    norm = np.sqrt(np.sum(direction**2 / masses))
    if norm < 1.0e-14:
        return None
    direction /= norm
    a = 0.5 * np.sum(direction**2 / masses)
    b = np.sum(momentum * direction / masses)
    c = target_energy - parent_energy
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    roots = [(-b + np.sqrt(discriminant)) / (2 * a), (-b - np.sqrt(discriminant)) / (2 * a)]
    alpha = min(roots, key=abs)
    return momentum + alpha * direction


def make_child(candidate: SpawnCandidate, parent: TrajectoryBasisFunction) -> TrajectoryBasisFunction | None:
    momentum = energy_matched_momentum(
        candidate.momenta,
        parent.masses,
        candidate.nac,
        float(candidate.energies[parent.state]),
        float(candidate.energies[candidate.target_state]),
    )
    if momentum is None:
        return None
    child = parent.copy_child(candidate.target_state, momentum)
    child.positions = candidate.positions.copy()
    child.time = candidate.time
    return child


def make_coupling_optimized_child(
    candidate: SpawnCandidate, parent: TrajectoryBasisFunction, seed: int
) -> TrajectoryBasisFunction | None:
    """Optional deterministic CMA refinement of a NAC energy-shell spawn."""
    try:
        import cma
    except ImportError as exc:
        raise RuntimeError("coupling_optimized spawning requires `pip install aims-berry[optimize]`") from exc
    baseline = energy_matched_momentum(
        candidate.momenta, parent.masses, candidate.nac,
        float(candidate.energies[parent.state]), float(candidate.energies[candidate.target_state]),
    )
    if baseline is None:
        return None
    target_kinetic = (
        np.sum(candidate.momenta**2 / (2 * parent.masses))
        + candidate.energies[parent.state] - candidate.energies[candidate.target_state]
    )
    if target_kinetic < 0:
        return None
    shape = baseline.shape

    def objective(flat):
        momentum = np.asarray(flat).reshape(shape)
        coupling = abs(np.sum(candidate.nac * momentum / parent.masses))
        kinetic_error = np.sum(momentum**2 / (2 * parent.masses)) - target_kinetic
        displacement = np.sum((momentum - baseline) ** 2 / parent.masses)
        return float(-coupling + 1.0e4 * kinetic_error**2 + 1.0e-6 * displacement)

    sigma = max(float(np.linalg.norm(baseline)) * 0.02, 1.0e-3)
    optimizer = cma.CMAEvolutionStrategy(
        baseline.reshape(-1), sigma,
        {"seed": int(seed), "maxiter": 60, "verb_disp": 0, "verbose": -9},
    )
    optimizer.optimize(objective)
    momentum = np.asarray(optimizer.result.xbest).reshape(shape)
    kinetic = np.sum(momentum**2 / (2 * parent.masses))
    if kinetic <= 0 and target_kinetic > 0:
        return None
    if kinetic > 0:
        momentum *= np.sqrt(target_kinetic / kinetic)
    child = parent.copy_child(candidate.target_state, momentum)
    child.positions = candidate.positions.copy()
    child.time = candidate.time
    return child


def overlaps_existing(child: TrajectoryBasisFunction, trajectories: list[TrajectoryBasisFunction], threshold: float) -> bool:
    return any(t.state == child.state and abs(gaussian_overlap(t, child, electronic=False)) > threshold for t in trajectories)


def prune_by_overlap(state: SimulationState, relative_threshold: float = 1.0e-8) -> list[int]:
    if state.matrices is None or len(state.trajectories) < 2:
        return []
    full_overlap = state.matrices.overlap
    keep = list(range(len(state.trajectories)))
    block = full_overlap.copy()
    values, vectors = np.linalg.eigh(0.5 * (block + block.conj().T))
    removed: list[int] = []
    while values[0] <= relative_threshold * max(values[-1], 1.0) and len(keep) > 1:
        mode = vectors[:, 0]
        local_drop = int(np.argmax(np.abs(mode)))
        removed.append(keep.pop(local_drop))
        block = full_overlap[np.ix_(keep, keep)]
        values, vectors = np.linalg.eigh(0.5 * (block + block.conj().T))
    if removed:
        state.trajectories = [state.trajectories[i] for i in keep]
        state.amplitudes = state.amplitudes[keep]
        state.matrices = None
    return sorted(removed)


class SpawnMonitor:
    """Detect threshold regions and emit candidates at observed local maxima."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.pending: dict[tuple[str, int], SpawnCandidate] = {}
        self.entries: dict[tuple[str, int], SpawnCandidate] = {}

    @staticmethod
    def key(candidate: SpawnCandidate) -> tuple[str, int]:
        return (candidate.parent_id or str(candidate.parent_index), candidate.target_state)

    def observe(self, candidate: SpawnCandidate) -> SpawnCandidate | None:
        key = self.key(candidate)
        previous = self.pending.get(key)
        if candidate.coupling >= self.threshold:
            if previous is None:
                self.entries[key] = candidate
            if previous is None or candidate.coupling > previous.coupling:
                self.pending[key] = candidate
            return None
        if previous is not None:
            del self.pending[key]
            entry = self.entries.pop(key)
            return dataclasses.replace(
                previous,
                entry_time=entry.time,
                entry_positions=entry.positions.copy(),
                entry_momenta=entry.momenta.copy(),
            )
        return None

    def flush(self) -> list[SpawnCandidate]:
        values = [
            dataclasses.replace(
                candidate,
                entry_time=self.entries[key].time,
                entry_positions=self.entries[key].positions.copy(),
                entry_momenta=self.entries[key].momenta.copy(),
            )
            for key, candidate in self.pending.items()
        ]
        self.pending.clear()
        self.entries.clear()
        return values

    def to_dict(self) -> dict:
        def encode(candidate):
            data = dataclasses.asdict(candidate)
            for key in ("positions", "momenta", "nac", "energies", "entry_positions", "entry_momenta"):
                value = data.get(key)
                if value is None:
                    continue
                array = np.asarray(value)
                data[key] = (
                    {"real": array.real.tolist(), "imag": array.imag.tolist()}
                    if np.iscomplexobj(array) else array.tolist()
                )
            return data
        return {
            "threshold": self.threshold,
            "pending": {f"{key[0]}:{key[1]}": encode(value) for key, value in self.pending.items()},
            "entries": {f"{key[0]}:{key[1]}": encode(value) for key, value in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpawnMonitor":
        monitor = cls(float(data["threshold"]))

        def decode(item):
            for key in ("positions", "momenta", "nac", "energies", "entry_positions", "entry_momenta"):
                if item.get(key) is not None:
                    value = item[key]
                    if isinstance(value, dict) and "real" in value:
                        item[key] = np.asarray(value["real"]) + 1j * np.asarray(value["imag"])
                    else:
                        item[key] = np.asarray(value)
            return SpawnCandidate(**item)

        for field in ("pending", "entries"):
            target = getattr(monitor, field)
            for key, value in data.get(field, {}).items():
                parent, state = key.rsplit(":", 1)
                target[(parent, int(state))] = decode(value)
        return monitor
