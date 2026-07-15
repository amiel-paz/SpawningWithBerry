from dataclasses import replace
import json
from pathlib import Path

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
from aims_berry.electronic.base import (
    ElectronicProperties,
    ElectronicStructureError,
    WavefunctionState,
)
from aims_berry.simulation import SimulationRunner
from aims_berry.spawning import SpawnCandidate


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


def test_mid_outer_step_failure_rolls_back_before_restart(tmp_path, monkeypatch):
    import aims_berry.simulation as simulation_module

    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nflat\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=0.05, simulation_time=0.15, random_seed=19,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        run_directory=tmp_path / "failed-step",
    )
    capabilities = ProviderCapabilities(nacs=True)
    original = simulation_module.adaptive_cayley_step
    injected = False

    def fail_once(*args, **kwargs):
        nonlocal injected
        if not injected:
            injected = True
            raise RuntimeError("injected outer-step failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation_module, "adaptive_cayley_step", fail_once)
    with np.testing.assert_raises_regex(RuntimeError, "injected outer-step failure"):
        run(config, CallableProvider(flat_provider, capabilities=capabilities))
    failed = json.loads(
        (config.run_directory / "checkpoint/current/checkpoint.json").read_text()
    )
    assert failed["runtime"]["active_step_snapshot"] is not None
    assert failed["trajectories"][0]["time"] == 0.05
    assert failed["quantum_time"] == 0.05
    with h5py.File(config.run_directory / "simulation.h5") as handle:
        assert sorted(handle["steps"]) == ["00000000"]

    monkeypatch.setattr(simulation_module, "adaptive_cayley_step", original)
    resumed = run(
        config, CallableProvider(flat_provider, capabilities=capabilities), restart=True
    )
    uninterrupted = run(
        replace(config, run_directory=tmp_path / "uninterrupted-step"),
        CallableProvider(flat_provider, capabilities=capabilities),
    )
    assert np.array_equal(resumed.state.amplitudes, uninterrupted.state.amplitudes)
    assert np.array_equal(resumed.state.matrices.hamiltonian, uninterrupted.state.matrices.hamiltonian)
    assert resumed.state.events == uninterrupted.state.events
    resumed_metadata = json.loads((resumed.checkpoint / "checkpoint.json").read_text())
    uninterrupted_metadata = json.loads(
        (uninterrupted.checkpoint / "checkpoint.json").read_text()
    )
    assert resumed_metadata["queue"] == uninterrupted_metadata["queue"]
    assert resumed_metadata["rng_state"] == uninterrupted_metadata["rng_state"]


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


def test_injected_replay_failure_restarts_from_transaction_entry(tmp_path, monkeypatch):
    import aims_berry.simulation as simulation_module

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
        norm_tolerance=1e-4, run_directory=tmp_path / "failed-replay",
    )
    capabilities = ProviderCapabilities(nacs=True)
    original = simulation_module.adaptive_cayley_step
    injected = False

    def fail_once(*args, **kwargs):
        nonlocal injected
        if not injected and args[1].overlap.shape == (2, 2):
            injected = True
            raise RuntimeError("injected replay failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation_module, "adaptive_cayley_step", fail_once)
    with np.testing.assert_raises_regex(RuntimeError, "injected replay failure"):
        run(config, CallableProvider(localized_coupling_provider, capabilities=capabilities))
    checkpoint = config.run_directory / "checkpoint/current/checkpoint.json"
    failed_metadata = json.loads(checkpoint.read_text())
    assert failed_metadata["step"] == 4
    assert failed_metadata["runtime"]["active_replay"] is not None
    with h5py.File(config.run_directory / "simulation.h5") as handle:
        assert int(handle["steps"].attrs["committed_through"]) == 4
        assert sorted(handle["replay"].keys())

    monkeypatch.setattr(simulation_module, "adaptive_cayley_step", original)
    resumed = run(
        config, CallableProvider(localized_coupling_provider, capabilities=capabilities),
        restart=True,
    )
    uninterrupted = run(
        replace(config, run_directory=tmp_path / "uninterrupted-replay"),
        CallableProvider(localized_coupling_provider, capabilities=capabilities),
    )
    assert np.allclose(resumed.state.amplitudes, uninterrupted.state.amplitudes, atol=1e-12)
    assert resumed.state.events == uninterrupted.state.events
    resumed_metadata = json.loads(
        (resumed.checkpoint / "checkpoint.json").read_text()
    )
    uninterrupted_metadata = json.loads(
        (uninterrupted.checkpoint / "checkpoint.json").read_text()
    )
    assert resumed_metadata["queue"] == uninterrupted_metadata["queue"]
    assert resumed_metadata["rng_state"] == uninterrupted_metadata["rng_state"]


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


def test_centroid_requests_only_the_required_properties(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nselective centroid\nH 0 0 0\n")
    requests = []

    def provider(request):
        requests.append(request)
        return flat_provider(request)

    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=1.0, simulation_time=1.0,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        run_directory=tmp_path / "selective-centroid",
    )
    runner = SimulationRunner(
        config, CallableProvider(provider, capabilities=ProviderCapabilities(nacs=True))
    )
    left = runner.state.trajectories[0]
    runner._evaluate_trajectory(0)
    assert requests[-1].gradient_states == (0,)
    assert requests[-1].nac_pairs == ((0, 1),)
    same_state = left.copy_child(0, left.momenta)
    runner._evaluate_centroid(left, same_state, np.zeros((1, 3)))
    assert ElectronicProperties.GRADIENTS not in requests[-1].properties
    assert ElectronicProperties.NACS not in requests[-1].properties
    other_state = left.copy_child(1, left.momenta)
    runner._evaluate_centroid(left, other_state, np.array([[0.1, 0.0, 0.0]]))
    assert requests[-1].nac_pairs == ((0, 1),)
    assert ElectronicProperties.GRADIENTS not in requests[-1].properties


def test_run_dataset_exposes_sparse_work_telemetry(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\ntelemetry\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=0.05, simulation_time=0.05,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        run_directory=tmp_path / "telemetry-run",
    )
    result = run(
        config,
        CallableProvider(flat_provider, capabilities=ProviderCapabilities(nacs=True)),
    )
    dataset = RunDataset(result.history)
    performance = dataset.performance_diagnostics()
    summary = dataset.summary()
    assert performance.shape == (2, 22)
    assert performance[-1, 5] == 2
    assert summary["maximum_population_sum_error"] < 1e-12
    assert summary["accepted_spawn_count"] == 0


def test_canonical_nuclear_step_selection_and_refinement(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nstep control\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=1, time_step=20.0, coupling_time_step=5.0,
        minimum_nuclear_time_step=0.625, simulation_time=40.0,
        electronic_method="custom", coupling_mode="nac", spawn_metric="nac_norm",
        spawn_threshold=3.0, run_directory=tmp_path / "steps",
    )
    runner = SimulationRunner(
        config, CallableProvider(flat_provider, capabilities=ProviderCapabilities(nacs=True))
    )
    nacs = np.zeros((2, 2, 1, 3), complex)
    nacs[0, 1, 0, 0] = 4.0
    nacs[1, 0] = -nacs[0, 1].conj()
    runner.state.trajectories[0].electronic = ElectronicStructureResult(
        energies=np.array([0.0, 0.01]), gradients=np.zeros((2, 1, 3)), nacs=nacs,
    ).validate(runner._request(np.zeros((1, 3)), 1, 0.0))
    assert runner._nuclear_time_step(40.0) == 5.0
    assert runner._refined_nuclear_time_step(20.0) == 5.0
    assert runner._refined_nuclear_time_step(5.0) == 2.5
    assert runner._refined_nuclear_time_step(1.25) == 0.625
    assert runner._refined_nuclear_time_step(0.625) is None
    assert runner._regularization_ladder() == (1.0e-8,)
    canonical = replace(config, regularization_threshold=1.0e-4)
    canonical_runner = SimulationRunner(
        canonical,
        CallableProvider(flat_provider, capabilities=ProviderCapabilities(nacs=True)),
    )
    assert canonical_runner._regularization_ladder() == (
        1.0e-4, 1.0e-5, 1.0e-6, 1.0e-8,
    )


def test_endpoint_energy_gradient_rejection_refines_complete_nuclear_interval(tmp_path):
    fixture = json.loads(
        (Path(__file__).parent / "data" / "ethylene_seed87063_continuity_failure.json")
        .read_text()
    )
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\ncontinuity refinement\nH 1 0 0\n")
    calls = []
    wavefunctions = {}

    def curvature_provider(request):
        previous_geometry = (
            None if request.previous is None else wavefunctions[request.previous.identifier]
        )
        displacement = (
            0.0
            if previous_geometry is None
            else float(np.linalg.norm(request.geometry - previous_geometry))
        )
        calls.append((request.time, displacement))
        if displacement > 0.05:
            raise ElectronicStructureError(
                "energy/gradient continuity failure: "
                f"residuals=[{fixture['reported_energy_gradient_residual_hartree']}], "
                f"tolerance={fixture['configured_tolerance_hartree']}"
            )
        identifier = f"wf-{len(calls)}"
        wavefunctions[identifier] = request.geometry.copy()
        return ElectronicStructureResult(
            energies=np.asarray([0.5 * request.geometry[0, 0] ** 2]),
            gradients=request.geometry[None, :, :].copy(),
            wavefunction=WavefunctionState(identifier),
        )

    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=1,
        initial_state=0, time_step=20.0, coupling_time_step=5.0,
        minimum_nuclear_time_step=0.625, simulation_time=20.0,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        energy_tolerance=0.005, run_directory=tmp_path / "continuity-refinement",
    )
    result = run(config, CallableProvider(curvature_provider))
    refinements = [
        event for event in result.events
        if event.get("kind") == "nuclear_step_refinement"
    ]
    assert result.state.quantum_time == 20.0
    assert refinements[0]["from_dt"] == 20.0
    assert refinements[0]["to_dt"] == 5.0
    assert "energy/gradient continuity failure" in refinements[0]["reason"]
    assert any(displacement > 0.05 for _time, displacement in calls)


def test_replay_history_is_hidden_until_transaction_commit(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nflat\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=0.05, simulation_time=0.1,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        run_directory=tmp_path / "transaction",
    )
    runner = SimulationRunner(
        config, CallableProvider(flat_provider, capabilities=ProviderCapabilities(nacs=True))
    )
    result = runner.propagate()
    runner.writer.begin_replay("test-window", entry_step=1, frontier_step=2)
    runner.writer.write_step(result.state, config.num_states, {"quantum_substeps": 8})
    assert [step["step"] for step in RunDataset(result.history).steps()] == [0, 1]
    with h5py.File(result.history) as handle:
        assert "00000002" in handle["replay/test-window/steps"]
        assert "00000002" not in handle["steps"]
        assert "00000002" in handle["replay/test-window/superseded_steps"]
        assert int(handle["steps"].attrs["committed_through"]) == 1
    runner.writer.commit_replay("test-window", 2)
    assert [step["step"] for step in RunDataset(result.history).steps()] == [0, 1, 2]
    with h5py.File(result.history) as handle:
        assert "test-window" not in handle["replay"]
        assert handle["steps/00000002"].attrs["quantum_substeps"] == 8


def test_entry_overlap_rejects_back_spawn_without_mutating_parent(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nflat\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=0.05, simulation_time=0.1,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        spawn_overlap_max=0.8, run_directory=tmp_path / "entry-overlap",
    )
    runner = SimulationRunner(
        config, CallableProvider(flat_provider, capabilities=ProviderCapabilities(nacs=True))
    )
    parent = runner.state.trajectories[0]
    parent.momenta[0, 0] = 2.0
    existing = parent.copy_child(1, parent.momenta)
    runner.state.add_trajectory(existing, 0j)
    runner._capture_spawn_snapshot((parent.identifier, 1))
    existing.positions[0, 0] = 10.0
    candidate = SpawnCandidate(
        0, 1, 0.2, 0.0, parent.positions.copy(), parent.momenta.copy(),
        np.array([[1.0, 0.0, 0.0]]), np.array([0.0, 0.0]),
        entry_time=0.0, parent_id=parent.identifier,
    )
    runner._spawn(candidate)
    assert parent.spawn_count == 0
    assert parent.last_spawn_time == -np.inf
    assert len(runner.state.trajectories) == 2
    assert runner.state.events[-1]["reason"] == "entry_overlap"
