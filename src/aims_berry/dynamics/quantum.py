"""Stable propagation in a nonorthogonal Gaussian basis."""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.integrate import solve_ivp

from ..core import MatrixSet


@dataclasses.dataclass(frozen=True)
class QuantumStepResult:
    """Accepted coefficients and diagnostics for one outer nuclear interval."""

    amplitudes: np.ndarray
    substeps: int
    norm_before: float
    norm_after_raw: float
    convergence_error: float
    metric_residual: float
    retained_overlap_eigenvalues: np.ndarray


class QuantumPropagationError(RuntimeError):
    """Raised when adaptive coefficient propagation cannot satisfy its gates."""

    def __init__(self, message: str, result: QuantumStepResult) -> None:
        super().__init__(message)
        self.result = result


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


def _interpolated_metric_matrices(
    start: MatrixSet,
    end: MatrixSet,
    dt: float,
    fraction: float,
) -> MatrixSet:
    """Interpolate endpoints while enforcing ``tau + tau**H == dS/dt``."""

    overlap = (1.0 - fraction) * start.overlap + fraction * end.overlap
    hamiltonian = (1.0 - fraction) * start.hamiltonian + fraction * end.hamiltonian
    raw_tau = (1.0 - fraction) * start.sdot + fraction * end.sdot
    metric_derivative = (end.overlap - start.overlap) / dt
    tau = 0.5 * (raw_tau - raw_tau.conj().T) + 0.5 * metric_derivative
    return MatrixSet(overlap, hamiltonian, tau)


def _phase_aligned_metric_error(
    coarse: np.ndarray,
    fine: np.ndarray,
    overlap: np.ndarray,
) -> float:
    inner = np.vdot(coarse, overlap @ fine)
    if abs(inner) < 1.0e-15:
        return float("inf")
    aligned_coarse = coarse * (inner / abs(inner))
    difference = fine - aligned_coarse
    return float(np.sqrt(max(metric_norm(difference, overlap), 0.0)))


def adaptive_cayley_step(
    amplitudes: np.ndarray,
    start: MatrixSet,
    end: MatrixSet,
    dt: float,
    *,
    threshold: float = 1.0e-8,
    convergence_tolerance: float = 1.0e-6,
    norm_tolerance: float = 1.0e-10,
    min_time_step: float | None = None,
) -> QuantumStepResult:
    """Adaptively converge generalized Cayley propagation between endpoints.

    Refinement changes only the coefficient solve and requests no midpoint
    electronic evaluations.  Material norm drift is rejected, never hidden by
    unconditional normalization.
    """

    if dt <= 0:
        raise ValueError("dt must be positive")
    if convergence_tolerance <= 0 or norm_tolerance <= 0:
        raise ValueError("adaptive tolerances must be positive")
    floor = dt / 4096.0 if min_time_step is None else float(min_time_step)
    if floor <= 0 or floor > dt:
        raise ValueError("min_time_step must be in (0, dt]")
    maximum = max(1, int(np.floor(dt / floor + 1.0e-12)))
    maximum_power = 1 << int(np.floor(np.log2(maximum)))

    coefficients = np.asarray(amplitudes, dtype=np.complex128)
    norm_before = metric_norm(coefficients, start.overlap)
    if norm_before <= 0:
        raise ValueError("cannot propagate a non-positive metric norm")
    raw_mid_tau = 0.5 * (start.sdot + end.sdot)
    metric_derivative = (end.overlap - start.overlap) / dt
    metric_residual = float(
        np.max(np.abs(metric_derivative - raw_mid_tau - raw_mid_tau.conj().T))
    )
    retained = _retained_subspace(end.overlap, threshold)[1]

    previous: np.ndarray | None = None
    last: QuantumStepResult | None = None
    substeps = 1
    while substeps <= maximum_power:
        propagated = coefficients.copy()
        sub_dt = dt / substeps
        for index in range(substeps):
            left = _interpolated_metric_matrices(start, end, dt, index / substeps)
            right = _interpolated_metric_matrices(
                start, end, dt, (index + 1) / substeps
            )
            propagated = cayley_step(propagated, left, sub_dt, threshold, right)
        norm_after = metric_norm(propagated, end.overlap)
        convergence_error = (
            float("inf")
            if previous is None
            else _phase_aligned_metric_error(previous, propagated, end.overlap)
        )
        last = QuantumStepResult(
            amplitudes=propagated,
            substeps=substeps,
            norm_before=norm_before,
            norm_after_raw=norm_after,
            convergence_error=convergence_error,
            metric_residual=metric_residual,
            retained_overlap_eigenvalues=retained.copy(),
        )
        if (
            previous is not None
            and convergence_error <= convergence_tolerance
            and abs(norm_after - norm_before) <= norm_tolerance
        ):
            accepted = propagated * np.sqrt(norm_before / norm_after)
            return dataclasses.replace(last, amplitudes=accepted)
        previous = propagated
        substeps *= 2

    assert last is not None
    raise QuantumPropagationError(
        "adaptive quantum propagation failed: "
        f"substeps={last.substeps}, convergence_error={last.convergence_error}, "
        f"norm_before={last.norm_before}, norm_after={last.norm_after_raw}, "
        f"metric_residual={last.metric_residual}",
        last,
    )


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
