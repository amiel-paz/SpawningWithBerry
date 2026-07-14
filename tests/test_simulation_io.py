from dataclasses import replace

import h5py
import numpy as np

from aims_berry import (
    CallableProvider,
    ElectronicStructureResult,
    ProviderCapabilities,
    SimulationConfig,
    run,
)
from aims_berry.analysis import RunDataset
from aims_berry.electronic.base import WavefunctionState
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


def localized_coupling_provider(request):
    coordinate = request.geometry[0, 0]
    coupling = 2.0 * np.exp(-(coordinate / 0.18) ** 2)
    nacs = np.zeros((2, 2, 1, 3), complex)
    nacs[1, 0, 0, 0] = coupling
    nacs[0, 1] = -nacs[1, 0].conj()
    return {
        "energies": np.array([0.0, 0.01]),
        "gradients": np.zeros((2, 1, 3)),
        "nacs": nacs,
    }


def test_spawn_rolls_back_and_replays_coupled_amplitudes_across_restart(tmp_path):
    xyz = tmp_path / "h.xyz"
    momenta = tmp_path / "momenta.txt"
    xyz.write_text("1\nlocalized coupling\nH -0.6 0 0\n")
    momenta.write_text("185 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=1, time_step=1.0, simulation_time=12.0,
        initial_condition="file", momenta=momenta, electronic_method="custom",
        coupling_mode="nac", spawn_threshold=0.03, population_to_spawn=0.05,
        spawn_cooldown=100.0, spawn_overlap_max=0.99, max_trajectories=2,
        norm_tolerance=1e-4, run_directory=tmp_path / "resumed",
    )
    capabilities = ProviderCapabilities(nacs=True)
    partial = SimulationRunner(
        config, CallableProvider(localized_coupling_provider, capabilities=capabilities)
    ).propagate(stop_after_step=5)
    assert partial.state.step == 5  # interrupted while the maximum is still being located

    resumed = run(
        config,
        CallableProvider(localized_coupling_provider, capabilities=capabilities),
        restart=True,
    )
    uninterrupted = run(
        replace(config, run_directory=tmp_path / "uninterrupted"),
        CallableProvider(localized_coupling_provider, capabilities=capabilities),
    )
    spawn = next(event for event in resumed.events if event.get("kind") == "spawn")
    assert spawn["threshold_entry_time"] == 4.0
    assert spawn["time"] == 6.0
    assert spawn["replay_frontier_time"] == 9.0
    assert len(resumed.state.trajectories) == 2
    assert resumed.state.populations(num_states=2)[0] > 1e-6
    assert [t.identifier for t in resumed.state.trajectories] == [
        t.identifier for t in uninterrupted.state.trajectories
    ]
    assert np.allclose(resumed.state.amplitudes, uninterrupted.state.amplitudes, atol=1e-12)
    with h5py.File(resumed.history) as handle:
        entry = handle["steps/00000004"]
        assert entry["amplitudes"].shape == (2,)
        assert entry["amplitudes"][1] == 0j


def test_endpoint_electronic_results_are_reused(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\ncounting\nH 0 0 0\n")
    calls = 0

    def counting_provider(request):
        nonlocal calls
        calls += 1
        return flat_provider(request)

    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=0.05, simulation_time=0.1,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        run_directory=tmp_path / "counting-run",
    )
    run(config, CallableProvider(counting_provider, capabilities=ProviderCapabilities(nacs=True)))
    assert calls == 3  # initial point plus one endpoint evaluation per step


def test_centroid_evaluations_reuse_pair_wavefunction_state(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\ncentroid tracking\nH 0 0 0\n")
    previous_ids = []

    def tracking_provider(request):
        previous_ids.append(request.previous.identifier if request.previous else None)
        nacs = np.zeros((2, 2, 1, 3), complex)
        nacs[0, 1, 0, 0] = 0.1
        nacs[1, 0] = -nacs[0, 1].conj()
        return ElectronicStructureResult(
            energies=np.array([0.0, 0.1]),
            gradients=(
                np.zeros((2, 1, 3))
                if request.properties and any(p.value == "gradients" for p in request.properties)
                else None
            ),
            nacs=nacs,
            wavefunction=WavefunctionState(f"wf-{len(previous_ids)}"),
        )

    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=0.1, simulation_time=0.1,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        run_directory=tmp_path / "tracking",
    )
    runner = SimulationRunner(
        config,
        CallableProvider(tracking_provider, capabilities=ProviderCapabilities(nacs=True)),
    )
    left = runner.state.trajectories[0]
    right = left.copy_child(1, left.momenta)
    runner._evaluate_centroid(left, right, np.zeros((1, 3)))
    runner.centroid_cache.clear()
    runner._evaluate_centroid(left, right, np.array([[0.1, 0.0, 0.0]]))
    assert previous_ids == [None, "wf-1"]
