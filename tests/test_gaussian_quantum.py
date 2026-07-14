import numpy as np
from scipy.linalg import expm

from aims_berry.core import MatrixSet, TrajectoryBasisFunction
from aims_berry.dynamics.gaussian import gaussian_kinetic, overlap_matrix
from aims_berry.dynamics.quantum import cayley_step, metric_norm, regularized_metric


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
