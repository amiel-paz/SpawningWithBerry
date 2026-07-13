import numpy as np

from aims_berry.core import MatrixSet, SimulationState, TrajectoryBasisFunction
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
