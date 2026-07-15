import json
from pathlib import Path

import numpy as np
import pytest

from aims_berry.config import AU_TIME_PER_FS, ConfigError, load_config
from aims_berry.geometry import masses_and_widths, read_xyz, sample_wigner


def _write(tmp_path: Path, extra: str = "") -> Path:
    (tmp_path / "quoted geometry.xyz").write_text("1\ncomment\nH 0 0 0\n")
    path = tmp_path / "aims.in"
    path.write_text(
        "provider custom # comment\n"
        'geometry "quoted geometry.xyz"\n'
        "num_states 2\ninitial_state 1\ntime_step 0.5 fs\nsimulation_time 2 au\n"
        "electronic_method custom\n"
        "provider_option label first\nprovider_option scale 2.5\n"
        "observable bond impossible_but_parsed 0 1\n" + extra
    )
    return path


def test_parser_comments_quoting_units_and_repeated_records(tmp_path):
    config = load_config(_write(tmp_path))
    assert config.geometry == tmp_path / "quoted geometry.xyz"
    assert config.time_step == pytest.approx(0.5 * AU_TIME_PER_FS)
    assert config.provider_option_dict() == {"label": "first", "scale": 2.5}
    assert config.observables[0].atoms == (0, 1)


@pytest.mark.parametrize("extra, match", [
    ("num_states 3\n", "duplicate"),
    ("not_a_keyword 4\n", "unknown keyword"),
    ("state_weights 0.2 banana\n", "could not convert"),
])
def test_parser_reports_filename_and_line(tmp_path, extra, match):
    path = _write(tmp_path, extra)
    with pytest.raises(ConfigError, match=match) as caught:
        load_config(path)
    assert str(path.resolve()) in str(caught.value)


def test_ethylene_production_config_and_wigner_sample_are_reproducible():
    root = Path(__file__).parents[1]
    config = load_config(root / "examples/ethylene_pyscf/production.in")
    atoms, equilibrium = read_xyz(config.geometry, config.geometry_units)
    masses, _ = masses_and_widths(atoms, config.gaussian_widths)
    first = sample_wigner(
        equilibrium,
        masses,
        config.hessian,
        np.random.default_rng(config.random_seed),
        config.temperature,
    )
    second = sample_wigner(
        equilibrium,
        masses,
        config.hessian,
        np.random.default_rng(config.random_seed),
        config.temperature,
    )
    assert config.num_states == 3
    assert config.state_weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert config.basis == "6-31g*"
    assert config.geometry.name == "ethylene.xyz"
    assert config.hessian.name == "ethylene_mp2_631gstar_hessian.txt"
    assert config.nsteps == 517
    assert config.time_step == 20.0
    assert config.coupling_time_step == 5.0
    assert config.minimum_nuclear_time_step == 0.625
    assert config.min_time_step == 0.00244140625
    assert config.spawn_metric == "nac_norm"
    assert config.spawn_threshold == 3.0
    assert config.pair_overlap_threshold == 1.0e-3
    assert config.nac_gap_threshold == pytest.approx(0.6 / 27.211386245988)
    assert config.provider_option_dict()["ci_root_overlap_min"] == 0.7
    assert config.simulation_time / AU_TIME_PER_FS == pytest.approx(249.9916951526321)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.all(np.isfinite(first[0]))
    assert np.all(np.isfinite(first[1]))
    hessian = np.loadtxt(config.hessian).reshape(equilibrium.size, equilibrium.size)
    inv_sqrt_mass = 1.0 / np.sqrt(masses)
    mass_weighted = hessian * inv_sqrt_mass[:, None] * inv_sqrt_mass[None, :]
    assert np.count_nonzero(np.linalg.eigvalsh(mass_weighted) > 1.0e-10) == 12


def test_ethylene_protocol_manifest_matches_production_controls():
    root = Path(__file__).parents[1]
    config = load_config(root / "examples/ethylene_pyscf/production.in")
    manifest = json.loads(
        (root / "examples/ethylene_pyscf/protocol-manifest.json").read_text()
    )
    canonical = manifest["canonical_aims_controls"]
    assert canonical["normal_time_step_au"] == config.time_step
    assert canonical["coupling_time_step_au"] == config.coupling_time_step
    assert canonical["spawn_threshold_bohr_inverse"] == config.spawn_threshold
    assert canonical["pair_overlap_threshold"] == config.pair_overlap_threshold
    assert manifest["reported_controls"]["electronic_method"].startswith(
        "equal-weight SA(3)-CAS(2e,2o)/6-31G*"
    )
