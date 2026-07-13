import numpy as np

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
