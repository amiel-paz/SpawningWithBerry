from dataclasses import replace

import h5py
import numpy as np

from aims_berry import CallableProvider, ProviderCapabilities, SimulationConfig, run
from aims_berry.analysis import RunDataset
from aims_berry.simulation import SimulationRunner


def flat_provider(request):
    nstate, natom = len(request.states), len(request.atoms)
    nac = np.zeros((nstate, nstate, natom, 3), complex)
    if nstate > 1:
        nac[0, 1, 0, 0] = 0.01
        nac[1, 0] = -nac[0, 1].conj()
    return {
        "energies": np.arange(nstate) * 0.05,
        "gradients": np.zeros((nstate, natom, 3)),
        "nacs": nac,
    }


def test_short_run_writes_history_and_exact_checkpoint(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nflat\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=0.05, simulation_time=0.1,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        run_directory=tmp_path / "run",
    )
    provider = CallableProvider(flat_provider, capabilities=ProviderCapabilities(nacs=True))
    result = run(config, provider)
    assert result.history.is_file()
    assert (result.checkpoint / "checkpoint.json").is_file()
    assert (result.checkpoint.parent / "previous" / "checkpoint.json").is_file()
    with h5py.File(result.history) as handle:
        assert sorted(handle["steps"]) == ["00000000", "00000001", "00000002"]
        assert handle["steps/00000002/amplitudes"].shape == (1,)
        assert handle["steps/00000002/nacs"].shape == (1, 2, 2, 1, 3)
    populations = RunDataset(result.history).populations()
    assert populations.shape == (3, 3)
    assert np.allclose(populations[:, 1:].sum(axis=1), 1.0, atol=1e-10)

    restarted = run(config, CallableProvider(flat_provider, capabilities=ProviderCapabilities(nacs=True)), restart=True)
    assert restarted.state.step == result.state.step
    assert [t.identifier for t in restarted.state.trajectories] == [t.identifier for t in result.state.trajectories]
    assert np.array_equal(restarted.state.amplitudes, result.state.amplitudes)
    assert np.array_equal(restarted.state.matrices.hamiltonian, result.state.matrices.hamiltonian)


def test_interrupted_restart_matches_uninterrupted_run(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nflat\nH 0 0 0\n")
    base = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=0.05, simulation_time=0.15, random_seed=77,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        run_directory=tmp_path / "resumed",
    )
    capabilities = ProviderCapabilities(nacs=True)
    runner = SimulationRunner(base, CallableProvider(flat_provider, capabilities=capabilities))
    partial = runner.propagate(stop_after_step=1)
    assert partial.state.step == 1
    resumed = run(base, CallableProvider(flat_provider, capabilities=capabilities), restart=True)
    uninterrupted = run(
        replace(base, run_directory=tmp_path / "uninterrupted"),
        CallableProvider(flat_provider, capabilities=capabilities),
    )
    assert [t.identifier for t in resumed.state.trajectories] == [t.identifier for t in uninterrupted.state.trajectories]
    assert np.array_equal(resumed.state.amplitudes, uninterrupted.state.amplitudes)
    assert np.array_equal(resumed.state.matrices.overlap, uninterrupted.state.matrices.overlap)
    assert RunDataset(resumed.history).populations().tolist() == RunDataset(uninterrupted.history).populations().tolist()
