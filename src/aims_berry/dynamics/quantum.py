"""Stable propagation in a nonorthogonal Gaussian basis."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..core import MatrixSet


def regularized_metric(overlap: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlap = 0.5 * (overlap + overlap.conj().T)
    values, vectors = np.linalg.eigh(overlap)
    cutoff = threshold * max(float(values.max()), 1.0)
    keep = values > cutoff
    if not np.any(keep):
        raise np.linalg.LinAlgError("all overlap eigenvalues were removed by regularization")
    vectors = vectors[:, keep]
    values = values[keep]
    s_half = (vectors * np.sqrt(values)) @ vectors.conj().T
    s_minus_half = (vectors / np.sqrt(values)) @ vectors.conj().T
    return s_half, s_minus_half, values


def _midpoint(start: MatrixSet, end: MatrixSet | None) -> MatrixSet:
    if end is None:
        return start
    if start.overlap.shape != end.overlap.shape:
        raise ValueError("start and end matrices must have the same shape")
    return MatrixSet(
        overlap=0.5 * (start.overlap + end.overlap),
        hamiltonian=0.5 * (start.hamiltonian + end.hamiltonian),
        sdot=0.5 * (start.sdot + end.sdot),
    )


def _retained_subspace(overlap: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    metric = 0.5 * (overlap + overlap.conj().T)
    values, vectors = np.linalg.eigh(metric)
    cutoff = threshold * max(float(values.max()), 1.0)
    keep = values > cutoff
    if not np.any(keep):
        raise np.linalg.LinAlgError("all overlap eigenvalues were removed by regularization")
    return vectors[:, keep], values[keep]


def cayley_step(
    amplitudes: np.ndarray,
    matrices: MatrixSet,
    dt: float,
    threshold: float = 1.0e-8,
    end_matrices: MatrixSet | None = None,
) -> np.ndarray:
    """Generalized midpoint Crank--Nicolson for ``S c_dot + tau c = -i H c``.

    A scalar energy reference is removed before the rational step and restored as a
    global phase.  This is essential for molecular Hamiltonians: otherwise a large,
    physically irrelevant electronic-energy offset suppresses interstate transfer.
    ``MatrixSet.sdot`` is the full right-acting derivative ``tau``, not ``dS/dt``.
    """

    midpoint = _midpoint(matrices, end_matrices)
    coefficients = np.asarray(amplitudes, dtype=np.complex128)
    metric_norm0 = np.vdot(coefficients, midpoint.overlap @ coefficients)
    if abs(metric_norm0) < 1.0e-14:
        raise ValueError("cannot propagate a zero-metric-norm coefficient vector")
    energy_reference = float(
        np.real(np.vdot(coefficients, midpoint.hamiltonian @ coefficients) / metric_norm0)
    )
    effective = midpoint.hamiltonian - energy_reference * midpoint.overlap - 1j * midpoint.sdot

    vectors, values = _retained_subspace(midpoint.overlap, threshold)
    projected_s = np.diag(values.astype(np.complex128))
    projected_k = vectors.conj().T @ effective @ vectors
    projected_c = vectors.conj().T @ coefficients
    left = projected_s + 0.5j * dt * projected_k
    right = projected_s - 0.5j * dt * projected_k
    propagated = np.linalg.solve(left, right @ projected_c)
    return np.exp(-1j * energy_reference * dt) * (vectors @ propagated)


def rk45_step(
    amplitudes: np.ndarray,
    matrices: MatrixSet,
    dt: float,
    threshold: float = 1.0e-8,
    end_matrices: MatrixSet | None = None,
) -> np.ndarray:
    midpoint = _midpoint(matrices, end_matrices)
    coefficients = np.asarray(amplitudes, dtype=np.complex128)
    norm = np.vdot(coefficients, midpoint.overlap @ coefficients)
    energy_reference = float(
        np.real(np.vdot(coefficients, midpoint.hamiltonian @ coefficients) / norm)
    )
    _, s_minus_half, _ = regularized_metric(midpoint.overlap, threshold)
    sinv = s_minus_half @ s_minus_half
    effective = midpoint.hamiltonian - energy_reference * midpoint.overlap - 1j * midpoint.sdot
    generator = -1j * sinv @ effective
    result = solve_ivp(lambda _t, y: generator @ y, (0.0, dt), coefficients, rtol=1e-9, atol=1e-11)
    if not result.success:
        raise RuntimeError(result.message)
    return np.exp(-1j * energy_reference * dt) * np.asarray(result.y[:, -1], dtype=np.complex128)


def metric_norm(amplitudes: np.ndarray, overlap: np.ndarray) -> float:
    return float(np.real(np.vdot(amplitudes, overlap @ amplitudes)))


def normalize(amplitudes: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    norm = metric_norm(amplitudes, overlap)
    if norm <= 0:
        raise ValueError("non-positive wavefunction norm")
    return amplitudes / np.sqrt(norm)
