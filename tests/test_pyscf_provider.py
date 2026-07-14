import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from aims_berry import load_config
from aims_berry.electronic.base import ElectronicProperties, ElectronicStructureRequest
from aims_berry.electronic.pyscf import from_config
from aims_berry.geometry import atomic_numbers, read_xyz


pytestmark = pytest.mark.pyscf


@pytest.mark.skipif(importlib.util.find_spec("pyscf") is None, reason="PySCF optional dependency is not installed")
def test_h3_sa_casscf_energies_gradients_and_nacs():
    root = Path(__file__).parents[1]
    config = load_config(root / "examples/h3_pyscf/aims.in")
    config = dataclasses.replace(
        config,
        provider_options=config.provider_options + (("orbital_selection", "overlap"),),
    )
    atoms, geometry = read_xyz(config.geometry, config.geometry_units)
    request = ElectronicStructureRequest(
        atoms=atoms, atomic_numbers=atomic_numbers(atoms), geometry=geometry,
        states=(0, 1), active_state=1, time=0.0,
        properties=frozenset({ElectronicProperties.ENERGIES, ElectronicProperties.GRADIENTS, ElectronicProperties.NACS}),
    )
    result = from_config(config).evaluate(request)
    assert result.energies[0] <= result.energies[1]
    assert result.gradients.shape == (2, 3, 3)
    assert result.nacs.shape == (2, 2, 3, 3)
    assert np.linalg.norm(result.nacs + result.nacs.swapaxes(0, 1).conj()) < 1e-8
    assert np.allclose(result.metadata["spin_squares"], [0.75, 0.75], atol=1e-8)

    tracked_request = ElectronicStructureRequest(
        atoms=atoms, atomic_numbers=atomic_numbers(atoms),
        geometry=geometry + np.asarray([[0.0, 1.0e-4, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        states=(0, 1), active_state=1, time=0.1,
        properties=request.properties, previous=result.wavefunction,
    )
    tracked = from_config(config)
    tracked.adopt_wavefunction(result.wavefunction)
    continued = tracked.evaluate(tracked_request)
    singular_values = continued.metadata["active_space_singular_values"]
    assert singular_values is not None and np.min(singular_values) > 0.99
    assert continued.metadata["energy_gradient_residuals"] is not None
