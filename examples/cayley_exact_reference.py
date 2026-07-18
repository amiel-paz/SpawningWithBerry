"""Print exact-reference errors for Cayley and PySpawn-style full diagonalization."""

from __future__ import annotations

import json

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from aims_berry.core import MatrixSet
from aims_berry.dynamics import adaptive_cayley_step, metric_norm


HAMILTONIAN = np.asarray([[0.08, 0.035], [0.035, -0.03]], complex)


def basis(time):
    theta, theta_dot = 0.025 * time + 0.0004 * time**2, 0.025 + 0.0008 * time
    exponents = np.asarray([0.20 * np.sin(0.07 * time), -0.15 * np.sin(0.05 * time)])
    dots = np.asarray([0.014 * np.cos(0.07 * time), -0.0075 * np.cos(0.05 * time)])
    rotation = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotation_dot = theta_dot * np.asarray(
        [[-np.sin(theta), -np.cos(theta)], [np.cos(theta), -np.sin(theta)]]
    )
    scaling = np.diag(np.exp(exponents))
    return rotation @ scaling, rotation_dot @ scaling + rotation @ np.diag(np.exp(exponents) * dots)


def matrices(time):
    frame, frame_dot = basis(time)
    return MatrixSet(frame.T @ frame, frame.T @ HAMILTONIAN @ frame, frame.T @ frame_dot)


def full_diagonal(coefficients, start, end, dt):
    result = coefficients.copy()
    for value in (start, end):
        effective = np.linalg.solve(value.overlap, value.hamiltonian - 1j * value.sdot)
        roots, vectors = np.linalg.eig(-0.5j * dt * effective)
        result = vectors @ (np.exp(roots) * np.linalg.solve(vectors, result))
    return result


def aligned_error(reference, trial, overlap):
    phase = np.vdot(reference, overlap @ trial)
    difference = trial * np.exp(-1j * np.angle(phase)) - reference
    return float(np.sqrt(max(metric_norm(difference, overlap), 0.0)))


def main():
    initial = np.asarray([1.0, 0.0], complex)
    fixed_matrices = MatrixSet(np.eye(2), HAMILTONIAN, np.zeros((2, 2)))
    fixed_duration = 10.0
    fixed_exact = expm(-1j * HAMILTONIAN * fixed_duration) @ initial
    fixed_cayley = adaptive_cayley_step(
        initial, fixed_matrices, fixed_matrices, fixed_duration,
        threshold=1.0e-12, convergence_tolerance=1.0e-7,
        norm_tolerance=1.0e-10, min_time_step=fixed_duration / 4096.0,
    )
    duration = 30.0
    initial_frame, _ = basis(0.0)
    final_frame, _ = basis(duration)
    exact = np.linalg.solve(
        final_frame,
        expm(-1j * HAMILTONIAN * duration) @ initial_frame @ initial,
    )
    final_metric = matrices(duration).overlap
    rows = []
    for dt in (2.0, 1.0, 0.5, 0.25):
        cayley, diagonal, time = initial.copy(), initial.copy(), 0.0
        maximum_substeps = 0
        while time < duration - 1.0e-12:
            start, end = matrices(time), matrices(time + dt)
            propagated = adaptive_cayley_step(
                cayley, start, end, dt,
                threshold=1.0e-12,
                convergence_tolerance=1.0e-7,
                norm_tolerance=1.0e-10,
                min_time_step=dt / 4096.0,
            )
            cayley = propagated.amplitudes
            maximum_substeps = max(maximum_substeps, propagated.substeps)
            diagonal = full_diagonal(diagonal, start, end, dt)
            time += dt
        rows.append({
            "outer_time_step_au": dt,
            "cayley_state_error": aligned_error(exact, cayley, final_metric),
            "full_diagonal_state_error": aligned_error(exact, diagonal, final_metric),
            "cayley_metric_norm_error": abs(metric_norm(cayley, final_metric) - 1.0),
            "full_diagonal_metric_norm_error": abs(metric_norm(diagonal, final_metric) - 1.0),
            "cayley_maximum_coefficient_substeps": maximum_substeps,
        })
    slope, coupling = 0.02, 0.04

    def crossing_hamiltonian(time):
        return np.asarray(
            [[slope * time, coupling], [coupling, -slope * time]], complex
        )

    crossing_initial = np.asarray([1.0, 0.0], complex)
    crossing_exact = solve_ivp(
        lambda time, coefficients: -1j * crossing_hamiltonian(time) @ coefficients,
        (-10.0, 10.0), crossing_initial,
        method="DOP853", rtol=1.0e-13, atol=1.0e-15,
    ).y[:, -1]
    crossing_cayley = crossing_initial.copy()
    crossing_diagonal = crossing_initial.copy()
    time, dt = -10.0, 2.0
    while time < 10.0 - 1.0e-12:
        start = MatrixSet(np.eye(2), crossing_hamiltonian(time), np.zeros((2, 2)))
        end = MatrixSet(np.eye(2), crossing_hamiltonian(time + dt), np.zeros((2, 2)))
        crossing_cayley = adaptive_cayley_step(
            crossing_cayley, start, end, dt,
            threshold=1.0e-12, convergence_tolerance=1.0e-7,
            norm_tolerance=1.0e-10, min_time_step=dt / 4096.0,
        ).amplitudes
        crossing_diagonal = full_diagonal(
            crossing_diagonal, start, end, dt
        )
        time += dt
    output = {
        "fixed_rabi_reference": {
            "description": "constant Hamiltonian; diagonalization is exact",
            "duration_au": fixed_duration,
            "cayley_state_error": aligned_error(
                fixed_exact, fixed_cayley.amplitudes, np.eye(2)
            ),
            "cayley_population_error": abs(
                abs(fixed_cayley.amplitudes[1]) ** 2 - abs(fixed_exact[1]) ** 2
            ),
            "cayley_metric_norm_error": abs(
                metric_norm(fixed_cayley.amplitudes, np.eye(2)) - 1.0
            ),
            "cayley_coefficient_substeps": fixed_cayley.substeps,
        },
        "moving_nonorthogonal_basis": rows,
        "landau_zener_avoided_crossing": {
            "outer_time_step_au": dt,
            "cayley_state_error": aligned_error(
                crossing_exact, crossing_cayley, np.eye(2)
            ),
            "full_diagonal_state_error": aligned_error(
                crossing_exact, crossing_diagonal, np.eye(2)
            ),
            "exact_final_state_1_population": float(abs(crossing_exact[1]) ** 2),
            "cayley_final_state_1_population": float(abs(crossing_cayley[1]) ** 2),
            "full_diagonal_final_state_1_population": float(abs(crossing_diagonal[1]) ** 2),
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
