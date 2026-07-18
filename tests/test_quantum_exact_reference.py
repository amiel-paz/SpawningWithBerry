"""Exact reference problems for moving-basis quantum propagation."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from aims_berry.core import MatrixSet
from aims_berry.dynamics import adaptive_cayley_step, metric_norm


PHYSICAL_HAMILTONIAN = np.asarray(
    [[0.08, 0.035], [0.035, -0.03]], dtype=np.complex128
)


def _phase_aligned_error(reference, trial, overlap):
    inner = np.vdot(reference, overlap @ trial)
    aligned = trial * np.exp(-1j * np.angle(inner))
    difference = aligned - reference
    return float(np.sqrt(max(metric_norm(difference, overlap), 0.0)))


def _basis(time):
    angle = 0.025 * time + 0.0004 * time**2
    angle_dot = 0.025 + 0.0008 * time
    exponents = np.asarray([
        0.20 * np.sin(0.07 * time),
        -0.15 * np.sin(0.05 * time),
    ])
    exponent_dots = np.asarray([
        0.014 * np.cos(0.07 * time),
        -0.0075 * np.cos(0.05 * time),
    ])
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    rotation_dot = angle_dot * np.asarray([
        [-np.sin(angle), -np.cos(angle)],
        [np.cos(angle), -np.sin(angle)],
    ])
    scaling = np.diag(np.exp(exponents))
    scaling_dot = np.diag(np.exp(exponents) * exponent_dots)
    return rotation @ scaling, rotation_dot @ scaling + rotation @ scaling_dot


def _moving_matrices(time):
    basis, basis_dot = _basis(time)
    return MatrixSet(
        basis.conj().T @ basis,
        basis.conj().T @ PHYSICAL_HAMILTONIAN @ basis,
        basis.conj().T @ basis_dot,
    )


def _exact_moving_coefficients(time, initial):
    initial_basis, _ = _basis(0.0)
    final_basis, _ = _basis(time)
    physical = expm(-1j * PHYSICAL_HAMILTONIAN * time) @ initial_basis @ initial
    return np.linalg.solve(final_basis, physical)


def _pyspawn_full_diagonal_step(coefficients, start, end, dt):
    result = coefficients.copy()
    for matrices in (start, end):
        effective = np.linalg.solve(
            matrices.overlap,
            matrices.hamiltonian - 1j * matrices.sdot,
        )
        eigenvalues, eigenvectors = np.linalg.eig(-0.5j * dt * effective)
        result = eigenvectors @ (
            np.exp(eigenvalues) * np.linalg.solve(eigenvectors, result)
        )
    return result


def test_cayley_crushes_exact_full_diagonal_rabi_reference():
    """For constant H, full diagonalization is the exact reference solution."""

    initial = np.asarray([1.0, 0.0], dtype=np.complex128)
    matrices = MatrixSet(
        np.eye(2), PHYSICAL_HAMILTONIAN, np.zeros((2, 2))
    )
    duration = 10.0
    exact = expm(-1j * PHYSICAL_HAMILTONIAN * duration) @ initial
    result = adaptive_cayley_step(
        initial,
        matrices,
        matrices,
        duration,
        threshold=1.0e-12,
        convergence_tolerance=1.0e-7,
        norm_tolerance=1.0e-10,
        min_time_step=duration / 4096.0,
    )

    assert result.substeps <= 2048
    assert abs(result.norm_after_raw - 1.0) < 1.0e-10
    assert _phase_aligned_error(exact, result.amplitudes, np.eye(2)) < 1.0e-8
    assert abs(abs(result.amplitudes[1]) ** 2 - abs(exact[1]) ** 2) < 1.0e-8


def test_moving_nonorthogonal_reference_converges_without_renormalization():
    """Cayley converges to a closed-form solution while preserving endpoint S norm."""

    initial = np.asarray([1.0, 0.0], dtype=np.complex128)
    duration = 30.0
    exact = _exact_moving_coefficients(duration, initial)
    final_metric = _moving_matrices(duration).overlap
    errors = []
    full_diagonal_norm_errors = []
    for dt in (2.0, 1.0, 0.5):
        cayley = initial.copy()
        full_diagonal = initial.copy()
        time = 0.0
        while time < duration - 1.0e-12:
            start = _moving_matrices(time)
            end = _moving_matrices(time + dt)
            result = adaptive_cayley_step(
                cayley,
                start,
                end,
                dt,
                threshold=1.0e-12,
                convergence_tolerance=1.0e-7,
                norm_tolerance=1.0e-10,
                min_time_step=dt / 4096.0,
            )
            cayley = result.amplitudes
            full_diagonal = _pyspawn_full_diagonal_step(
                full_diagonal, start, end, dt
            )
            time += dt
        errors.append(_phase_aligned_error(exact, cayley, final_metric))
        assert abs(metric_norm(cayley, final_metric) - 1.0) < 1.0e-9
        full_diagonal_norm_errors.append(
            abs(metric_norm(full_diagonal, final_metric) - 1.0)
        )

    # Endpoint interpolation is second order: halving the outer step should
    # reduce the exact physical-state error by approximately four.
    assert errors[1] < errors[0] / 3.5
    assert errors[2] < errors[1] / 3.5
    assert full_diagonal_norm_errors[2] < full_diagonal_norm_errors[1]
    assert full_diagonal_norm_errors[1] < full_diagonal_norm_errors[0]


def test_adaptive_cayley_crushes_endpoint_diagonalization_at_avoided_crossing():
    """Adaptive time ordering resolves a standard Landau--Zener crossing."""

    slope, coupling = 0.02, 0.04

    def hamiltonian(time):
        return np.asarray(
            [[slope * time, coupling], [coupling, -slope * time]],
            dtype=np.complex128,
        )

    initial = np.asarray([1.0, 0.0], dtype=np.complex128)
    start_time, end_time, dt = -10.0, 10.0, 2.0
    exact = solve_ivp(
        lambda time, coefficients: -1j * hamiltonian(time) @ coefficients,
        (start_time, end_time),
        initial,
        method="DOP853",
        rtol=1.0e-13,
        atol=1.0e-15,
    ).y[:, -1]
    cayley, full_diagonal, time = initial.copy(), initial.copy(), start_time
    while time < end_time - 1.0e-12:
        start = MatrixSet(np.eye(2), hamiltonian(time), np.zeros((2, 2)))
        end = MatrixSet(np.eye(2), hamiltonian(time + dt), np.zeros((2, 2)))
        cayley = adaptive_cayley_step(
            cayley,
            start,
            end,
            dt,
            threshold=1.0e-12,
            convergence_tolerance=1.0e-7,
            norm_tolerance=1.0e-10,
            min_time_step=dt / 4096.0,
        ).amplitudes
        full_diagonal = _pyspawn_full_diagonal_step(
            full_diagonal, start, end, dt
        )
        time += dt

    cayley_error = _phase_aligned_error(exact, cayley, np.eye(2))
    full_diagonal_error = _phase_aligned_error(
        exact, full_diagonal, np.eye(2)
    )
    assert cayley_error < 2.0e-7
    assert full_diagonal_error > 1.0e-3
    assert cayley_error < full_diagonal_error / 10_000.0
    assert abs(abs(cayley[1]) ** 2 - abs(exact[1]) ** 2) < 1.0e-7
    assert abs(metric_norm(cayley, np.eye(2)) - 1.0) < 1.0e-10
