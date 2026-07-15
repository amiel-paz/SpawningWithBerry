import numpy as np

from aims_berry.core import MatrixSet, SimulationState, TrajectoryBasisFunction
from aims_berry.dynamics.gaussian import gaussian_momentum
from aims_berry.dynamics.hamiltonian import SaddlePointHamiltonian
from aims_berry.electronic.base import ElectronicStructureResult
from aims_berry.spawning import SpawnCandidate, energy_matched_momentum, make_child, prune_by_overlap
from aims_berry.tasks import TaskKind, TaskQueue


def test_spawn_momentum_is_on_energy_shell():
    momentum = np.array([[2.0, 0.0, 0.0]])
    masses = np.array([[2.0, 2.0, 2.0]])
    direction = np.array([[1.0, 0.0, 0.0]])
    adjusted = energy_matched_momentum(momentum, masses, direction, 0.0, 0.4)
    assert adjusted is not None
    before = np.sum(momentum**2 / (2 * masses))
    after = np.sum(adjusted**2 / (2 * masses)) + 0.4
    assert abs(before - after) < 1e-12


def test_child_creation_is_same_position_and_deterministically_labeled():
    parent = TrajectoryBasisFunction(np.zeros((1, 3)), np.array([[2.0, 0, 0]]), np.ones((1, 3)), np.full((1, 3), 2.0), 0)
    candidate = SpawnCandidate(0, 1, 0.2, 3.0, parent.positions.copy(), parent.momenta.copy(), np.array([[1.0, 0, 0]]), np.array([0.0, 0.2]))
    child = make_child(candidate, parent)
    assert child is not None and np.array_equal(child.positions, parent.positions)
    assert child.label == "00b1" and child.parent_id == parent.identifier
    assert parent.spawn_count == 0
    assert parent.last_spawn_time == -np.inf


def test_task_queue_restart_preserves_dependencies_and_order():
    queue = TaskQueue()
    early = queue.add(TaskKind.TRAJECTORY, 0.0, "early")
    late = queue.add(TaskKind.QUANTUM, 0.0, "late", dependencies=[early.identifier])
    assert queue.pop_ready().identifier == "early"
    queue.complete(early)
    restored = TaskQueue.from_dict(queue.to_dict())
    assert restored.pop_ready().identifier == late.identifier


def test_near_linearly_dependent_basis_is_pruned():
    first = TrajectoryBasisFunction(np.zeros((1, 3)), np.zeros((1, 3)), np.ones((1, 3)), np.ones((1, 3)), 0)
    second = TrajectoryBasisFunction(np.full((1, 3), 1e-8), np.zeros((1, 3)), np.ones((1, 3)), np.ones((1, 3)), 0)
    overlap = np.array([[1.0, 1.0 - 1e-15], [1.0 - 1e-15, 1.0]], complex)
    state = SimulationState([first, second], np.array([1.0, 0.0]), matrices=MatrixSet(overlap, np.eye(2), np.zeros((2, 2))))
    removed = prune_by_overlap(state, 1e-8)
    assert len(removed) == 1 and len(state.trajectories) == 1


def test_nac_saddle_point_element_has_derivative_operator_sign():
    left = TrajectoryBasisFunction(
        np.array([[-0.1, 0.0, 0.0]]), np.array([[0.4, 0.0, 0.0]]),
        np.ones((1, 3)), np.full((1, 3), 2.0), 0,
    )
    right = TrajectoryBasisFunction(
        np.array([[0.2, 0.0, 0.0]]), np.array([[0.7, 0.0, 0.0]]),
        np.ones((1, 3)), np.full((1, 3), 2.0), 1,
    )
    nac = np.zeros((2, 2, 1, 3), complex)
    nac[0, 1, 0, 0] = 0.3
    nac[1, 0] = -nac[0, 1].conj()
    result = ElectronicStructureResult(
        energies=np.array([0.0, 0.1]), gradients=np.zeros((2, 1, 3)), nacs=nac
    )
    left.electronic = result
    right.electronic = result
    calls = []

    def centroid(a, b, geometry):
        calls.append((a.identifier, b.identifier, geometry.copy()))
        return result

    matrices = SaddlePointHamiltonian("nac").build(
        [left, right], centroid,
        [left.momenta / left.masses, right.momenta / right.masses],
        [np.zeros((1, 3)), np.zeros((1, 3))], 0.1,
    )
    expected = 1j * np.sum(nac[0, 1] * gaussian_momentum(left, right) / right.masses)
    assert np.allclose(matrices.hamiltonian[0, 1], expected)
    assert np.allclose(matrices.hamiltonian[1, 0], expected.conjugate())
    assert len(calls) == 1  # diagonal electronic data are reused from the TBFs


def test_pair_overlap_screening_issues_no_centroid_call():
    left = TrajectoryBasisFunction(
        np.zeros((1, 3)), np.zeros((1, 3)), np.ones((1, 3)), np.ones((1, 3)), 0,
    )
    right = TrajectoryBasisFunction(
        np.full((1, 3), 10.0), np.zeros((1, 3)), np.ones((1, 3)), np.ones((1, 3)), 1,
    )
    electronic = ElectronicStructureResult(
        energies=np.array([0.0, 0.1]), gradients=np.zeros((2, 1, 3)),
    )
    left.electronic = right.electronic = electronic

    def forbidden_centroid(*_args):
        raise AssertionError("screened TBF pair requested a centroid")

    matrices = SaddlePointHamiltonian("nac", pair_overlap_threshold=1.0e-3).build(
        [left, right], forbidden_centroid,
        [np.zeros((1, 3)), np.zeros((1, 3))],
        [np.zeros((1, 3)), np.zeros((1, 3))], 1.0,
    )
    assert matrices.overlap[0, 1] == 0
    assert matrices.hamiltonian[0, 1] == 0
    assert matrices.sdot[0, 1] == 0
