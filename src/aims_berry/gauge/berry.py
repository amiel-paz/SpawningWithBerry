"""Electronic-state tracking and geometric-phase utilities."""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.linalg import polar
from scipy.optimize import linear_sum_assignment


@dataclasses.dataclass
class GaugeTransform:
    permutation: np.ndarray
    unitary: np.ndarray
    aligned_overlap: np.ndarray


class GaugeTracker:
    """Parallel-transport an adiabatic state manifold between geometries."""

    def align(self, overlap: np.ndarray) -> GaugeTransform:
        overlap = np.asarray(overlap, dtype=np.complex128)
        if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
            raise ValueError("state overlap must be square")
        row, col = linear_sum_assignment(-np.abs(overlap))
        permutation = col[np.argsort(row)]
        reordered = overlap[:, permutation]
        unitary, _positive = polar(reordered.conj().T)
        aligned = reordered @ unitary
        return GaugeTransform(permutation=permutation, unitary=unitary, aligned_overlap=aligned)


def transform_nacs(nacs: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    """Apply an electronic unitary to both state indices of a NAC tensor."""
    return np.einsum("ai,abxy,bj->ijxy", unitary.conj(), nacs, unitary, optimize=True)


def wilson_loop(overlaps: list[np.ndarray], *, subspace: tuple[int, ...] | None = None) -> float:
    if not overlaps:
        return 0.0
    size = overlaps[0].shape[0]
    indices = np.arange(size) if subspace is None else np.asarray(subspace, dtype=int)
    transport = np.eye(len(indices), dtype=np.complex128)
    for overlap in overlaps:
        block = np.asarray(overlap)[np.ix_(indices, indices)]
        unitary, _ = polar(block)
        transport = transport @ unitary
    return float(np.angle(np.linalg.det(transport)))


def line_integral_phase(points: np.ndarray, connection) -> float:
    """Midpoint line integral of a vector-valued Berry connection."""
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        return 0.0
    total = 0.0
    for start, end in zip(points[:-1], points[1:]):
        midpoint = 0.5 * (start + end)
        total += float(np.real(np.dot(connection(midpoint), end - start)))
    return float(np.angle(np.exp(1j * total)))


def phase_distance(value: float, target: float) -> float:
    return float(abs(np.angle(np.exp(1j * (value - target)))))
