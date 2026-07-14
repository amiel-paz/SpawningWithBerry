from pathlib import Path

import numpy as np
from scipy.linalg import expm

from aims_berry.core import MatrixSet, TrajectoryBasisFunction
from aims_berry.dynamics.gaussian import gaussian_kinetic, gaussian_sdot, overlap_matrix
from aims_berry.dynamics.quantum import (
    adaptive_cayley_step,
    cayley_step,
    metric_norm,
    regularized_metric,
)


def trajectory(x, p):
    return TrajectoryBasisFunction(
        positions=np.array([[x, 0.0, 0.0]]), momenta=np.array([[p, 0.0, 0.0]]),
        widths=np.full((1, 3), 2.0), masses=np.full((1, 3), 1837.0), state=0,
    )


def test_gaussian_matrices_are_hermitian_and_metric_positive():
    basis = [trajectory(-0.1, 0.2), trajectory(0.15, -0.1)]
    overlap = overlap_matrix(basis)
    kinetic = np.array([[gaussian_kinetic(a, b) for b in basis] for a in basis])
    assert np.linalg.norm(overlap - overlap.conj().T) < 1e-12
    assert np.linalg.norm(kinetic - kinetic.conj().T) < 1e-12
    _, _, retained = regularized_metric(overlap, 1e-10)
    assert np.all(retained > 0)


def test_metric_cayley_conserves_norm_to_acceptance_threshold():
    overlap = np.array([[1.0, 0.2j], [-0.2j, 1.0]], complex)
    hamiltonian = np.array([[0.1, 0.03 + 0.02j], [0.03 - 0.02j, 0.2]])
    matrices = MatrixSet(overlap, hamiltonian, np.zeros((2, 2), complex))
    coefficients = np.array([1.0, 0.25j])
    coefficients /= np.sqrt(metric_norm(coefficients, overlap))
    initial = metric_norm(coefficients, overlap)
    for _ in range(100):
        coefficients = cayley_step(coefficients, matrices, 0.01)
    assert abs(metric_norm(coefficients, overlap) - initial) < 1e-10


def test_cayley_reproduces_archived_ethylene_transfer_and_ignores_energy_offset():
    # Step 500 of the invalidated run: the former unshifted Cayley step produced
    # |c_child|=4.7e-7 even though the stored interstate element is 2.36e-3 Eh.
    hamiltonian = np.array([
        [-77.6156504, -0.0007257589600039746 - 0.0022423493191650336j],
        [-0.0007257589600039746 + 0.0022423493191650336j, -77.6159233],
    ])
    overlap = np.eye(2, dtype=complex)
    coefficients = np.array([
        -0.965655093 + 0.259827331j,
        -3.16851078e-7 + 1.04121634e-8j,
    ])
    matrices = MatrixSet(overlap, hamiltonian, np.zeros((2, 2), complex))
    propagated = cayley_step(coefficients, matrices, 10.0)
    exact = expm(-10j * hamiltonian) @ coefficients
    assert abs(abs(propagated[1]) - 0.023566) < 2e-6
    assert abs(abs(propagated[1]) - abs(exact[1])) < 2e-6

    offset = 1000.0
    shifted = MatrixSet(overlap, hamiltonian + offset * overlap, np.zeros((2, 2), complex))
    shifted_result = cayley_step(coefficients, shifted, 10.0)
    assert np.allclose(np.abs(shifted_result) ** 2, np.abs(propagated) ** 2, atol=1e-12)


def test_cayley_uses_full_right_acting_derivative():
    overlap = np.eye(2, dtype=complex)
    hamiltonian = np.array([[0.1, 0.02], [0.02, 0.15]], complex)
    tau = np.array([[0.0, 0.01], [-0.01, 0.0]], complex)
    coefficients = np.array([1.0, 0.0j])
    with_tau = cayley_step(coefficients, MatrixSet(overlap, hamiltonian, tau), 0.01)
    without_tau = cayley_step(
        coefficients, MatrixSet(overlap, hamiltonian, np.zeros_like(tau)), 0.01
    )
    exact = expm(-0.01j * (hamiltonian - 1j * tau)) @ coefficients
    assert not np.allclose(with_tau, without_tau)
    assert np.allclose(np.abs(with_tau), np.abs(exact), atol=1e-10)
    assert abs(metric_norm(with_tau, overlap) - 1.0) < 1e-12


def test_adaptive_cayley_repairs_three_tbf_ethylene_failure():
    fixture = np.load(Path(__file__).parent / "data" / "ethylene_second_spawn_step.npz")
    start = MatrixSet(fixture["start_S"], fixture["start_H"], fixture["start_tau"])
    end = MatrixSet(fixture["end_S"], fixture["end_H"], fixture["end_tau"])
    coefficients = fixture["amplitudes"]

    old = cayley_step(coefficients, start, 10.0, end_matrices=end)
    assert abs(metric_norm(old, end.overlap) - 1.000063744080636) < 2.0e-12

    repaired = adaptive_cayley_step(
        coefficients,
        start,
        end,
        10.0,
        convergence_tolerance=1.0e-6,
        norm_tolerance=1.0e-10,
        min_time_step=10.0 / 4096.0,
    )
    assert repaired.substeps <= 4096
    assert abs(repaired.norm_after_raw - repaired.norm_before) < 1.0e-10
    assert abs(metric_norm(repaired.amplitudes, end.overlap) - repaired.norm_before) < 2.0e-14

    reference = adaptive_cayley_step(
        coefficients,
        start,
        end,
        10.0,
        convergence_tolerance=2.5e-7,
        norm_tolerance=1.0e-11,
        min_time_step=10.0 / 4096.0,
    )
    assert np.allclose(
        np.abs(repaired.amplitudes) ** 2,
        np.abs(reference.amplitudes) ** 2,
        atol=1.0e-6,
    )

    shifted_start = MatrixSet(
        start.overlap,
        start.hamiltonian + 1000.0 * start.overlap,
        start.sdot,
    )
    shifted_end = MatrixSet(
        end.overlap,
        end.hamiltonian + 1000.0 * end.overlap,
        end.sdot,
    )
    shifted = adaptive_cayley_step(
        coefficients,
        shifted_start,
        shifted_end,
        10.0,
        convergence_tolerance=1.0e-6,
        norm_tolerance=1.0e-10,
        min_time_step=10.0 / 4096.0,
    )
    assert np.allclose(
        np.abs(shifted.amplitudes) ** 2,
        np.abs(repaired.amplitudes) ** 2,
        atol=1.0e-12,
    )


def test_gaussian_tau_matches_finite_difference_metric_derivative():
    basis = [trajectory(-0.12, 0.3), trajectory(0.17, -0.2)]
    velocities = [np.array([[0.04, 0.0, 0.0]]), np.array([[-0.03, 0.0, 0.0]])]
    momentum_derivatives = [
        np.array([[0.02, 0.0, 0.0]]), np.array([[-0.01, 0.0, 0.0]])
    ]
    tau = np.asarray([
        [
            gaussian_sdot(left, right, velocities[j], momentum_derivatives[j])
            for j, right in enumerate(basis)
        ]
        for left in basis
    ])
    epsilon = 1.0e-6
    displaced = []
    for sign in (-1.0, 1.0):
        current = []
        for item, velocity, pdot in zip(basis, velocities, momentum_derivatives):
            moved = trajectory(
                item.positions[0, 0] + sign * epsilon * velocity[0, 0],
                item.momenta[0, 0] + sign * epsilon * pdot[0, 0],
            )
            current.append(moved)
        displaced.append(overlap_matrix(current))
    finite_difference = (displaced[1] - displaced[0]) / (2.0 * epsilon)
    assert np.max(np.abs(finite_difference - tau - tau.conj().T)) < 1.0e-9
