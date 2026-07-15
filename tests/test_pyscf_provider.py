import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from aims_berry import load_config
from aims_berry.electronic.base import (
    ElectronicProperties,
    ElectronicStructureError,
    ElectronicStructureRequest,
)
from aims_berry.electronic.pyscf import _energy_ordered_root_phases, from_config
from aims_berry.geometry import atomic_numbers, read_xyz


pytestmark = pytest.mark.pyscf


def test_ethylene_overlap_assignment_is_diagnostic_not_an_adiabatic_permutation():
    fixture = json.loads(
        (Path(__file__).parent / "data" / "ethylene_seed87063_continuity_failure.json")
        .read_text()
    )
    overlap = np.asarray(fixture["rejected_tracking_overlap"])
    phases, assignment, diagonal = _energy_ordered_root_phases(overlap, 0.0)
    assert assignment.tolist() == fixture["rejected_assignment_suggestion"]
    assert np.argmin(fixture["rejected_raw_energy_order_hartree"]) == 0
    assert np.allclose(np.abs(phases), 1.0)
    assert diagonal[1] == pytest.approx(0.09290320)
    with pytest.raises(ElectronicStructureError, match="CI-root continuity failure"):
        _energy_ordered_root_phases(overlap, 0.7)

    # The raw CI arrays live in different active-orbital gauges.  Once the
    # determinant overlap includes that orbital transformation, all three
    # physical energy-ordered roots are continuous and comfortably pass the gate.
    proper = np.asarray(fixture["orbital_aware_root_overlap"])
    _phases, proper_assignment, proper_diagonal = _energy_ordered_root_phases(
        proper, 0.7
    )
    assert proper_assignment.tolist() == fixture["orbital_aware_assignment"]
    assert np.min(proper_diagonal) > 0.99
    assert abs(fixture["corrected_active_state_energy_gradient_residual_hartree"]) < (
        fixture["configured_tolerance_hartree"]
    )


@pytest.mark.skipif(importlib.util.find_spec("pyscf") is None, reason="PySCF optional dependency is not installed")
def test_active_orbital_alignment_contragrediently_transforms_ci_coefficients():
    from pyscf.fci import addons

    theta = 0.83
    orbital_rotation = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    ci_old = np.asarray([[0.71, 0.17], [-0.31, 0.61]])
    ci_old /= np.linalg.norm(ci_old)

    # Represent the same CAS wavefunction first in a rotated active basis and
    # then parallel-transport that basis back to the original gauge.
    ci_rotated = addons.transform_ci_for_orbital_rotation(
        ci_old, 2, (1, 1), orbital_rotation
    )
    ci_aligned = addons.transform_ci_for_orbital_rotation(
        ci_rotated, 2, (1, 1), orbital_rotation.T
    )
    assert np.allclose(ci_aligned, ci_old, atol=1.0e-12)


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


@pytest.mark.skipif(importlib.util.find_spec("pyscf") is None, reason="PySCF optional dependency is not installed")
def test_h3_selective_gradient_and_nac_match_full_request():
    root = Path(__file__).parents[1]
    config = load_config(root / "examples/h3_pyscf/aims.in")
    atoms, geometry = read_xyz(config.geometry, config.geometry_units)
    common = dict(
        atoms=atoms, atomic_numbers=atomic_numbers(atoms), geometry=geometry,
        states=(0, 1), active_state=1, time=0.0,
        properties=frozenset({
            ElectronicProperties.ENERGIES,
            ElectronicProperties.GRADIENTS,
            ElectronicProperties.NACS,
        }),
    )
    provider = from_config(config)
    full = provider.evaluate(ElectronicStructureRequest(**common))
    selective_request = ElectronicStructureRequest(
        **common, previous=full.wavefunction, gradient_states=(1,),
        nac_pairs=((0, 1),), nac_gap_threshold=1.0,
    )
    selective = provider.evaluate(selective_request)
    assert selective.gradient_mask.tolist() == [False, True]
    assert selective.nac_mask.tolist() == [[False, True], [True, False]]
    assert np.allclose(selective.energies, full.energies, atol=1e-9)
    assert np.allclose(selective.gradient_at_state(1), full.gradient_at_state(1), atol=1e-8)
    assert np.allclose(selective.nac_between(0, 1), full.nac_between(0, 1), atol=1e-8)
