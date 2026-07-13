import numpy as np

from aims_berry.electronic.base import ElectronicProperties, ElectronicStructureRequest
from aims_berry.gauge.berry import GaugeTracker, line_integral_phase, phase_distance, transform_nacs, wilson_loop
from aims_berry.models import BerryModel2DParallelTransport


def test_parallel_transport_and_nac_gauge_covariance():
    rng = np.random.default_rng(14)
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, 2))
    overlap = np.diag(phases)
    transform = GaugeTracker().align(overlap)
    assert np.allclose(transform.aligned_overlap, np.eye(2), atol=1e-12)
    nac = np.zeros((2, 2, 1, 3), complex)
    nac[0, 1, 0, 0] = 0.2 + 0.4j
    nac[1, 0] = -nac[0, 1].conj()
    rotated = transform_nacs(nac, np.diag(phases))
    assert np.allclose(rotated + rotated.swapaxes(0, 1).conj(), 0)


def test_analytic_closed_loop_has_pi_berry_phase():
    model = BerryModel2DParallelTransport(phase_gradient=5.0)
    height = np.pi / model.phase_gradient
    points = np.array([[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, height, 0.0], [-4.0, height, 0.0], [-4.0, 0.0, 0.0]])
    phase = line_integral_phase(points, lambda point: model.connection(point.reshape(1, 3), 0).reshape(-1))
    assert phase_distance(phase, np.pi) < 1e-2


def test_wilson_loop_is_invariant_to_intermediate_gauges():
    berry = np.exp(0.35j)
    overlaps = [np.diag([np.exp(0.2j), 1]), np.diag([np.exp(0.15j), 1])]
    gauges = [np.diag(np.exp(1j * np.array([0.0, 0.0]))), np.diag(np.exp(1j * np.array([1.1, -0.3]))), np.eye(2)]
    transformed = [gauges[i].conj().T @ overlaps[i] @ gauges[i + 1] for i in range(2)]
    assert phase_distance(wilson_loop(overlaps), np.angle(berry)) < 1e-12
    assert phase_distance(wilson_loop(transformed), np.angle(berry)) < 1e-12


def test_berry_provider_returns_complex_antihermitian_couplings():
    provider = BerryModel2DParallelTransport()
    request = ElectronicStructureRequest(
        atoms=("H",), atomic_numbers=np.array([1]), geometry=np.array([[0.1, 0.2, 0.0]]),
        states=(0, 1), active_state=0, time=0,
        properties=frozenset({ElectronicProperties.ENERGIES, ElectronicProperties.GRADIENTS, ElectronicProperties.NACS}),
    )
    result = provider.evaluate(request)
    assert np.iscomplexobj(result.nacs)
    assert np.linalg.norm(result.nacs + result.nacs.swapaxes(0, 1).conj()) < 1e-12
