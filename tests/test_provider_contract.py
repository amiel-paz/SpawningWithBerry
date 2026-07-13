import numpy as np
import pytest

from aims_berry import CallableProvider, ProviderCapabilities
from aims_berry.electronic.base import (
    ElectronicProperties,
    ElectronicStructureError,
    ElectronicStructureRequest,
)


def request():
    return ElectronicStructureRequest(
        atoms=("H",), atomic_numbers=np.array([1]), geometry=np.zeros((1, 3)),
        states=(0, 1), active_state=0, time=0.0,
        properties=frozenset({ElectronicProperties.ENERGIES, ElectronicProperties.GRADIENTS, ElectronicProperties.NACS}),
    )


def test_home_baked_callable_needs_no_core_changes():
    def calculator(_request):
        nac = np.zeros((2, 2, 1, 3), complex)
        nac[0, 1, 0, 0] = 0.4 + 0.2j
        nac[1, 0] = -nac[0, 1].conj()
        return {"energies": [0.0, 0.1], "gradients": np.zeros((2, 1, 3)), "nacs": nac}

    provider = CallableProvider(calculator, capabilities=ProviderCapabilities(nacs=True))
    result = provider.evaluate(request())
    assert result.nacs.shape == (2, 2, 1, 3)
    assert np.allclose(result.nacs + result.nacs.swapaxes(0, 1).conj(), 0)


def test_contract_rejects_non_antihermitian_nacs():
    def calculator(_request):
        return {"energies": [0.0, 0.1], "gradients": np.zeros((2, 1, 3)), "nacs": np.ones((2, 2, 1, 3))}

    provider = CallableProvider(calculator, capabilities=ProviderCapabilities(nacs=True))
    with pytest.raises(ElectronicStructureError, match="anti-Hermitian"):
        provider.evaluate(request())
