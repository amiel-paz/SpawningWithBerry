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


def cayley_step(amplitudes: np.ndarray, matrices: MatrixSet, dt: float, threshold: float = 1.0e-8) -> np.ndarray:
    s_half, s_minus_half, _ = regularized_metric(matrices.overlap, threshold)
    effective = matrices.hamiltonian - 0.5j * (matrices.sdot + matrices.sdot.conj().T)
    transformed = s_minus_half.conj().T @ effective @ s_minus_half
    transformed = 0.5 * (transformed + transformed.conj().T)
    identity = np.eye(transformed.shape[0], dtype=np.complex128)
    y0 = s_half @ np.asarray(amplitudes, dtype=np.complex128)
    y1 = np.linalg.solve(identity + 0.5j * dt * transformed, (identity - 0.5j * dt * transformed) @ y0)
    return s_minus_half @ y1


def rk45_step(amplitudes: np.ndarray, matrices: MatrixSet, dt: float, threshold: float = 1.0e-8) -> np.ndarray:
    _, s_minus_half, _ = regularized_metric(matrices.overlap, threshold)
    sinv = s_minus_half @ s_minus_half
    generator = -1j * sinv @ matrices.effective
    result = solve_ivp(lambda _t, y: generator @ y, (0.0, dt), amplitudes, rtol=1e-9, atol=1e-11)
    if not result.success:
        raise RuntimeError(result.message)
    return np.asarray(result.y[:, -1], dtype=np.complex128)


def metric_norm(amplitudes: np.ndarray, overlap: np.ndarray) -> float:
    return float(np.real(np.vdot(amplitudes, overlap @ amplitudes)))


def normalize(amplitudes: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    norm = metric_norm(amplitudes, overlap)
    if norm <= 0:
        raise ValueError("non-positive wavefunction norm")
    return amplitudes / np.sqrt(norm)
