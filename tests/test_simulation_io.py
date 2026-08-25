from dataclasses import replace
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from aims_berry import (
    CallableProvider,
    ElectronicStructureResult,
    ProviderCapabilities,
    SimulationConfig,
    __version__,
    run,
)
from aims_berry.analysis import RunDataset
from aims_berry.core import MatrixSet
from aims_berry.electronic.base import (
    ElectronicProperties,
    ElectronicStructureError,
    WavefunctionState,
)
from aims_berry.simulation import SimulationRunner, restart_from_checkpoint
from aims_berry.models.pyspawn_cone import PySpawnTestCone
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


def harmonic_provider(request):
    geometry = np.asarray(request.geometry)
    return {
        "energies": np.asarray([0.5 * geometry[0, 0] ** 2]),
        "gradients": geometry[None, :, :].copy(),
    }


def test_certified_npi_transport_is_not_polar_aligned_away(tmp_path):
    xyz = tmp_path / "cone.xyz"
    xyz.write_text("1\ncone\nH 0.45 0.10 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=1, time_step=0.1, simulation_time=0.0,
        electronic_method="analytic_model", coupling_mode="npi",
        spawn_metric="tdc", spawn_momentum="isotropic",
        run_directory=tmp_path / "cone-run",
    )
    runner = SimulationRunner(config, PySpawnTestCone())
    first = runner._evaluate(np.asarray([[0.45, 0.10, 0.0]]), 1, 0.0)
    second = runner._evaluate(
        np.asarray([[0.44, 0.11, 0.0]]), 1, 0.1, first.wavefunction
    )
    assert second.metadata["gauge_transport"] == "provider_certified_npi"
    assert abs(second.state_overlaps[0, 1]) > 1.0e-4
    assert abs(runner._npi_tdc(second)[0, 1]) > 1.0e-3
    runner.close()


def test_checkpoint_restart_can_extend_simulation_endpoint(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nextended restart\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=1,
        initial_state=0, time_step=0.1, simulation_time=0.2,
        electronic_method="custom", run_directory=tmp_path / "extended-restart",
    )
    provider = CallableProvider(flat_provider)
    initial = run(config, provider)
    assert initial.state.quantum_time == pytest.approx(0.2)

    extended = restart_from_checkpoint(
        initial.checkpoint,
        CallableProvider(flat_provider),
        simulation_time=0.4,
    )

    assert extended.state.quantum_time == pytest.approx(0.4)
    metadata = json.loads((extended.checkpoint / "checkpoint.json").read_text())
    assert metadata["config"]["simulation_time"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="no earlier"):
        restart_from_checkpoint(
            extended.checkpoint,
            CallableProvider(flat_provider),
            simulation_time=0.3,
        )


def test_optional_xyz_history_export_uses_angstrom_and_committed_frames(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nxyz export\nH 1 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=1,
        initial_state=0, time_step=0.1, simulation_time=0.0,
        electronic_method="custom", write_xyz=True,
        run_directory=tmp_path / "xyz-run",
    )
    result = run(config, CallableProvider(flat_provider))
    exported = result.run_directory / "geometries/step-00000000/tbf-0000-00.xyz"
    lines = exported.read_text().splitlines()
    assert lines[0] == "1"
    assert "step=0" in lines[1]
    assert "state=0" in lines[1]
    assert float(lines[2].split()[1]) == pytest.approx(1 / 1.8897261254578281)
    with h5py.File(result.history) as handle:
        assert [value.decode() if isinstance(value, bytes) else value for value in handle["atoms"][:]] == ["H"]


def test_classical_energy_tolerance_can_be_stricter_than_quantum_gate(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nclassical gate\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=1,
        initial_state=0, time_step=1.0, simulation_time=0.0,
        electronic_method="custom", energy_tolerance=5.0e-3,
        classical_energy_tolerance=2.0e-4,
        classical_energy_numerical_margin=1.0e-8,
        run_directory=tmp_path / "classical-gate",
    )
    runner = SimulationRunner(config, CallableProvider(flat_provider))
    trajectory = runner.state.trajectories[0]
    trajectory.electronic = ElectronicStructureResult(
        energies=np.asarray([0.0]), gradients=np.zeros((1, 1, 3)),
    )
    runner.classical_energy_references[trajectory.identifier] = 0.0
    trajectory.momenta[0, 0] = np.sqrt(
        2.0 * trajectory.masses[0, 0] * 3.0e-4
    )
    with pytest.raises(RuntimeError, match="classical energy violation") as caught:
        runner._check_classical_energies()
    assert "kinetic=" in str(caught.value)
    assert "potential=" in str(caught.value)
    trajectory.momenta[0, 0] = np.sqrt(
        2.0 * trajectory.masses[0, 0] * (2.0e-4 + 0.5e-8)
    )
    runner._check_classical_energies()
    runner.close()


def test_classical_energy_record_policy_logs_and_continues(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nclassical record policy\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=1,
        initial_state=0, time_step=1.0, simulation_time=0.0,
        electronic_method="custom", classical_energy_tolerance=2.0e-4,
        classical_energy_policy="record",
        run_directory=tmp_path / "classical-record",
    )
    runner = SimulationRunner(config, CallableProvider(flat_provider))
    trajectory = runner.state.trajectories[0]
    trajectory.electronic = ElectronicStructureResult(
        energies=np.asarray([0.0]), gradients=np.zeros((1, 1, 3)),
    )
    runner.classical_energy_references[trajectory.identifier] = 0.0
    trajectory.momenta[0, 0] = np.sqrt(
        2.0 * trajectory.masses[0, 0] * 3.0e-4
    )
    runner._check_classical_energies()
    runner._check_classical_energies()
    events = [
        event for event in runner.state.events
        if event.get("kind") == "classical_energy_threshold"
    ]
    assert len(events) == 1
    assert events[0]["policy"] == "record"
    assert runner.classical_energy_gate_active[trajectory.identifier]
    snapshot = runner._current_snapshot((trajectory.identifier, 0))
    runner.classical_energy_gate_active.clear()
    runner._restore_spawn_snapshot(snapshot)
    assert runner.classical_energy_gate_active[trajectory.identifier]
    runner.close()


def test_classical_energy_violation_retries_velocity_verlet_transactionally(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nadaptive classical step\nH 1 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=1,
        initial_state=0, time_step=1.0, minimum_nuclear_time_step=0.25,
        simulation_time=1.0, electronic_method="custom",
        nuclear_masses=(1.0, 1.0e30, 1.0e30),
        classical_energy_tolerance=1.0e-3,
        run_directory=tmp_path / "adaptive-classical",
    )
    runner = SimulationRunner(config, CallableProvider(harmonic_provider))
    result = runner.propagate(stop_after_step=1)
    refinements = [
        event for event in result.events
        if event.get("kind") == "nuclear_step_refinement"
    ]
    assert result.state.quantum_time == pytest.approx(0.25)
    assert [(event["from_dt"], event["to_dt"]) for event in refinements] == [
        (1.0, 0.5), (0.5, 0.25),
    ]
    assert all("classical energy violation" in event["reason"] for event in refinements)
    assert result.state.step == 1
    with h5py.File(result.history) as handle:
        assert sorted(handle["steps"], key=int) == ["00000000", "00000001"]
        endpoint = handle["steps/00000001"]
        assert endpoint.attrs["nuclear_time_step"] == pytest.approx(0.25)
        assert endpoint.attrs["quantum_energy_reference"] == pytest.approx(
            handle["steps/00000000"].attrs["quantum_energy"]
        )
        assert endpoint.attrs["quantum_energy_drift"] == pytest.approx(
            endpoint.attrs["quantum_energy"]
            - endpoint.attrs["quantum_energy_reference"]
        )
        assert endpoint["classical_reference_energy"][0] == pytest.approx(0.5)
        assert endpoint["classical_energy_drift"][0] == pytest.approx(
            endpoint["classical_total_energy"][0] - 0.5
        )
        assert endpoint["classical_potential_energy"][0] == pytest.approx(
            endpoint["energies"][0, endpoint["states"][0]]
        )
    absolute = RunDataset(result.history).absolute_energies()["00"]
    assert absolute.shape == (2, 8)
    assert absolute[0, 2] == pytest.approx(0.5)
    assert absolute[-1, -2] == pytest.approx(0.5)
    assert absolute[-1, -1] == pytest.approx(absolute[-1, -3] - 0.5)
    runner.close()


def test_classical_energy_adaptation_stops_with_diagnostic_at_floor(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nadaptive floor\nH 1 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=1,
        initial_state=0, time_step=1.0, minimum_nuclear_time_step=0.5,
        simulation_time=1.0, electronic_method="custom",
        nuclear_masses=(1.0, 1.0e30, 1.0e30),
        classical_energy_tolerance=1.0e-3,
        run_directory=tmp_path / "adaptive-floor",
    )
    runner = SimulationRunner(config, CallableProvider(harmonic_provider))
    with pytest.raises(RuntimeError, match="classical energy violation"):
        runner.propagate()
    checkpoint = json.loads(
        (config.run_directory / "checkpoint/current/checkpoint.json").read_text()
    )
    assert checkpoint["runtime"]["active_step_snapshot"] is not None
    assert checkpoint["runtime"]["forced_nuclear_time_steps"] == {
        "0.000000000000": 0.5
    }
    runner.close()


def test_quantum_energy_policy_records_or_rejects_without_changing_amplitudes(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nquantum energy policy\nH 0 0 0\n")
    base = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=1,
        initial_state=0, time_step=1.0, simulation_time=0.0,
        electronic_method="custom", energy_tolerance=1.0e-3,
        run_directory=tmp_path / "quantum-record",
    )
    runner = SimulationRunner(base, CallableProvider(flat_provider))
    runner.state.matrices = MatrixSet(np.eye(1), np.zeros((1, 1)), np.zeros((1, 1)))
    runner._check_quantum_energy()
    before = runner.state.amplitudes.copy()
    runner.state.matrices = MatrixSet(
        np.eye(1), np.asarray([[2.0e-3]]), np.zeros((1, 1))
    )
    runner._check_quantum_energy()
    assert np.array_equal(runner.state.amplitudes, before)
    assert runner.state.events[-1]["kind"] == "quantum_energy_threshold"
    assert runner.state.events[-1]["policy"] == "record"
    runner.close()

    strict = SimulationRunner(
        replace(
            base,
            quantum_energy_policy="error",
            run_directory=tmp_path / "quantum-error",
        ),
        CallableProvider(flat_provider),
    )
    strict.state.matrices = MatrixSet(np.eye(1), np.zeros((1, 1)), np.zeros((1, 1)))
    strict._check_quantum_energy()
    strict.state.matrices = MatrixSet(
        np.eye(1), np.asarray([[2.0e-3]]), np.zeros((1, 1))
    )
    with pytest.raises(RuntimeError, match="quantum energy violation"):
        strict._check_quantum_energy()
    strict.close()


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
    checkpoint_metadata = json.loads((result.checkpoint / "checkpoint.json").read_text())
    assert checkpoint_metadata["version_info"]["version"] == __version__
    assert checkpoint_metadata["version_info"]["last_update"]
    with h5py.File(result.history) as handle:
        assert sorted(handle["steps"]) == ["00000000", "00000001", "00000002"]
        assert handle["steps/00000002/amplitudes"].shape == (1,)
        assert handle["steps/00000002/nacs"].shape == (1, 2, 2, 1, 3)
    readable = result.run_directory / "readable"
    with (readable / "populations.csv").open(newline="") as stream:
        population_rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in population_rows] == [0, 1, 2]
    assert float(population_rows[-1]["state_0"]) == pytest.approx(1.0)
    with (readable / "quantum_diagnostics.csv").open(newline="") as stream:
        diagnostic_rows = list(csv.DictReader(stream))
    assert float(diagnostic_rows[-1]["metric_norm"]) == pytest.approx(1.0)
    tbf_directory = next(path for path in (readable / "tbfs").iterdir() if path.is_dir())
    assert (tbf_directory / "energies.csv").is_file()
    assert (tbf_directory / "phase_space.csv").is_file()
    assert (tbf_directory / "couplings.csv").is_file()
    assert (tbf_directory / "derivative_norms.csv").is_file()
    with (tbf_directory / "phase_space.csv").open(newline="") as stream:
        phase_rows = list(csv.DictReader(stream))
    assert float(phase_rows[-1]["gross_tbf_population"]) == pytest.approx(1.0)
    status = json.loads((readable / "run_status.json").read_text())
    assert status["last_step"] == 2
    assert status["populations"] == pytest.approx([1.0, 0.0])
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
    assert spawn["threshold_entry_time"] == 3.0
    assert spawn["time"] == 6.0
    assert spawn["replay_frontier_time"] == 9.0
    assert len(resumed.state.trajectories) == 2
    assert resumed.state.populations(num_states=2)[0] > 1e-6
    assert [t.identifier for t in resumed.state.trajectories] == [
        t.identifier for t in uninterrupted.state.trajectories
    ]
    assert np.allclose(resumed.state.amplitudes, uninterrupted.state.amplitudes, atol=1e-12)
    with h5py.File(resumed.history) as handle:
        entry = handle["steps/00000003"]
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
    assert failed_metadata["step"] == 3
    assert failed_metadata["runtime"]["active_replay"] is not None
    with h5py.File(config.run_directory / "simulation.h5") as handle:
        assert int(handle["steps"].attrs["committed_through"]) == 3
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
    assert runner._refined_nuclear_time_step(20.0) == 10.0
    assert runner._refined_nuclear_time_step(10.0) == 5.0
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
    runner.forced_nuclear_time_steps = {"5.000000000000": 1.25}
    runner.replay_suppressed = {(runner.state.trajectories[0].identifier, 0): 7.5}
    runner.quantum_energy_gate_active = True
    runner.centroid_nac_previous = {"pair": np.asarray([[1.0, 2.0j]])}
    snapshot = runner._current_snapshot((runner.state.trajectories[0].identifier, 0))
    runner.forced_nuclear_time_steps["5.000000000000"] = 0.625
    runner.replay_suppressed.clear()
    runner.quantum_energy_gate_active = False
    runner.centroid_nac_previous.clear()
    runner._restore_spawn_snapshot(snapshot)
    assert runner.forced_nuclear_time_steps == {"5.000000000000": 1.25}
    assert runner.replay_suppressed == {
        (runner.state.trajectories[0].identifier, 0): 7.5
    }
    assert runner.quantum_energy_gate_active
    assert np.array_equal(
        runner.centroid_nac_previous["pair"], np.asarray([[1.0, 2.0j]])
    )
    runner.close()
    canonical_runner.close()


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
    assert refinements[0]["to_dt"] == 10.0
    assert "energy/gradient continuity failure" in refinements[0]["reason"]
    assert any(displacement > 0.05 for _time, displacement in calls)


def test_spawn_child_backprop_refines_rejected_interval_transactionally(tmp_path):
    fixture = json.loads(
        (Path(__file__).parent / "data" / "ethylene_seed87063_spawn_backprop_failure.json")
        .read_text()
    )
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nspawn backprop refinement\nH 0 0 0\n")
    calls = []
    wavefunction_times = {}

    def continuity_provider(request):
        previous_time = (
            None
            if request.previous is None
            else wavefunction_times[request.previous.identifier]
        )
        interval = 0.0 if previous_time is None else abs(request.time - previous_time)
        calls.append((request.time, interval))
        if interval > 1.0 + 1.0e-12:
            raise ElectronicStructureError(
                "energy/gradient continuity failure: "
                f"residuals=[{fixture['reported_energy_gradient_residual_hartree']}], "
                f"tolerance={fixture['configured_tolerance_hartree']}"
            )
        identifier = f"wf-{len(calls)}"
        wavefunction_times[identifier] = request.time
        return ElectronicStructureResult(
            energies=np.asarray([0.0]),
            gradients=np.zeros((1, 1, 3)),
            wavefunction=WavefunctionState(identifier),
        )

    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=1,
        initial_state=0, time_step=2.0, coupling_time_step=2.0,
        minimum_nuclear_time_step=1.0, simulation_time=4.0,
        electronic_method="custom", spawn_threshold=1e9,
        run_directory=tmp_path / "spawn-backprop-refinement",
    )
    runner = SimulationRunner(config, CallableProvider(continuity_provider))
    child = runner.state.trajectories[0].copy_child(0, np.zeros((1, 3)))
    child.time = 4.0
    propagated = runner._backpropagate_spawn_child(child, 0.0)

    refinements = [
        event for event in runner.state.events
        if event.get("kind") == "spawn_backprop_refinement"
    ]
    assert propagated.time == pytest.approx(0.0)
    assert all(event["from_dt"] == 2.0 for event in refinements)
    assert all(event["to_dt"] == 1.0 for event in refinements)
    assert len(refinements) == 3
    assert sum(time == 4.0 for time, _interval in calls) == 1
    assert any(interval == 2.0 for _time, interval in calls)
    assert len(runner.state.trajectories) == 1
    runner.close()


def test_endpoint_centroid_continuity_rejection_refines_complete_interval(
    tmp_path, monkeypatch
):
    fixture = json.loads(
        (Path(__file__).parent / "data" / "ethylene_seed87063_spawn_backprop_failure.json")
        .read_text()
    )
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\ncentroid refinement\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=1,
        initial_state=0, time_step=2.0, coupling_time_step=2.0,
        minimum_nuclear_time_step=1.0, simulation_time=2.0,
        electronic_method="custom", spawn_threshold=1e9,
        run_directory=tmp_path / "centroid-refinement",
    )
    runner = SimulationRunner(config, CallableProvider(flat_provider))
    original_build = runner._build_matrices
    calls = 0

    def reject_first_endpoint(dt=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            overlaps = fixture["centroid_root_diagonal_overlaps"]
            raise ElectronicStructureError(
                "PySCF CI-root continuity failure: "
                f"diagonal_overlaps={overlaps}, assignment_suggestion=[1, 0, 2], "
                f"minimum={fixture['centroid_root_overlap_minimum']}"
            )
        return original_build(dt)

    monkeypatch.setattr(runner, "_build_matrices", reject_first_endpoint)
    result = runner.propagate()
    refinements = [
        event for event in result.events
        if event.get("kind") == "nuclear_step_refinement"
    ]
    assert result.state.quantum_time == 2.0
    assert refinements[0]["from_dt"] == 2.0
    assert refinements[0]["to_dt"] == 1.0
    assert "CI-root continuity failure" in refinements[0]["reason"]
    runner.close()


def test_centroid_nac_parallel_transport_removes_provider_phase_flips(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\ncentroid NAC phase\nH 0 0 0\n")
    calls = 0

    def phase_flipping_provider(request):
        nonlocal calls
        calls += 1
        sign = 1.0 if calls % 2 else -1.0
        nacs = np.zeros((2, 2, 1, 3), complex)
        nacs[0, 1, 0, 0] = sign
        nacs[1, 0] = -nacs[0, 1].conj()
        return ElectronicStructureResult(
            energies=np.asarray([0.0, 0.01]),
            nacs=nacs,
        )

    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=1.0, simulation_time=0.0,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        run_directory=tmp_path / "centroid-nac-phase",
    )
    runner = SimulationRunner(
        config,
        CallableProvider(
            phase_flipping_provider,
            capabilities=ProviderCapabilities(nacs=True),
        ),
    )
    left = runner.state.trajectories[0]
    right = left.copy_child(1, left.momenta)
    first = runner._evaluate_centroid(left, right, np.zeros((1, 3)))
    second = runner._evaluate_centroid(
        left, right, np.asarray([[0.1, 0.0, 0.0]])
    )
    assert np.allclose(first.nac_between(0, 1), second.nac_between(0, 1))
    assert np.allclose(
        second.metadata["centroid_nac_parallel_transport_phase"], -1.0
    )
    runner.close()


def test_centroid_nac_phase_tracking_skips_gap_screened_pair(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nscreened centroid NAC\nH 0 0 0\n")

    def screened_provider(request):
        return ElectronicStructureResult(
            energies=np.asarray([0.0, 0.1]),
            nacs=np.zeros((2, 2, 1, 3), complex),
            nac_mask=np.zeros((2, 2), bool),
        )

    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=1.0, simulation_time=0.0,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        nac_gap_threshold=0.01,
        run_directory=tmp_path / "screened-centroid-nac",
    )
    runner = SimulationRunner(
        config,
        CallableProvider(
            screened_provider, capabilities=ProviderCapabilities(nacs=True)
        ),
    )
    left = runner.state.trajectories[0]
    right = left.copy_child(1, left.momenta)
    result = runner._evaluate_centroid(left, right, np.zeros((1, 3)))
    assert not result.nac_mask[0, 1]
    assert runner.centroid_nac_previous == {}
    runner.close()


def test_gap_screening_closes_active_spawn_window(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nspawn screen boundary\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=1, time_step=1.0, simulation_time=0.0,
        electronic_method="custom", coupling_mode="nac",
        spawn_metric="projected", spawn_threshold=0.01,
        max_energy_gap=0.01, run_directory=tmp_path / "spawn-screen-boundary",
    )
    runner = SimulationRunner(
        config,
        CallableProvider(flat_provider, capabilities=ProviderCapabilities(nacs=True)),
    )
    trajectory = runner.state.trajectories[0]
    trajectory.momenta[:] = trajectory.masses
    nacs = np.zeros((2, 2, 1, 3), complex)
    nacs[1, 0, 0, 0] = 0.02
    nacs[0, 1] = -nacs[1, 0].conj()
    trajectory.electronic = ElectronicStructureResult(
        energies=np.array([0.0, 0.005]), nacs=nacs,
        nac_mask=np.array([[False, True], [True, False]]),
    )
    runner.state.matrices = MatrixSet(
        np.eye(1), np.zeros((1, 1)), np.zeros((1, 1))
    )
    assert runner._observe_spawning() == []
    key = (trajectory.identifier, 0)
    assert key in runner.monitor.pending
    assert runner.active_spawn_snapshot is not None

    trajectory.electronic = ElectronicStructureResult(
        energies=np.array([0.0, 0.02]), nacs=nacs,
        nac_mask=np.zeros((2, 2), bool),
    )
    completed = runner._observe_spawning()

    assert len(completed) == 1
    assert completed[0].target_state == 0
    assert completed[0].coupling == pytest.approx(0.02)
    assert key not in runner.monitor.pending
    runner.close()


def test_deferred_replay_candidate_blocks_duplicate_window(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\ndeferred spawn\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=1, time_step=1.0, simulation_time=0.0,
        electronic_method="custom", coupling_mode="nac",
        spawn_metric="projected", spawn_threshold=0.01,
        run_directory=tmp_path / "deferred-spawn",
    )
    runner = SimulationRunner(
        config,
        CallableProvider(flat_provider, capabilities=ProviderCapabilities(nacs=True)),
    )
    trajectory = runner.state.trajectories[0]
    trajectory.momenta[:] = trajectory.masses
    nacs = np.zeros((2, 2, 1, 3), complex)
    nacs[1, 0, 0, 0] = 0.02
    nacs[0, 1] = -nacs[1, 0].conj()
    trajectory.electronic = ElectronicStructureResult(
        energies=np.array([0.0, 0.005]), nacs=nacs,
        nac_mask=np.array([[False, True], [True, False]]),
    )
    runner.state.matrices = MatrixSet(
        np.eye(1), np.zeros((1, 1)), np.zeros((1, 1))
    )
    runner.deferred_spawns.append(
        SpawnCandidate(
            0, 0, 0.02, 0.0, trajectory.positions.copy(),
            trajectory.momenta.copy(), nacs[1, 0].copy(),
            trajectory.electronic.energies.copy(),
            parent_id=trajectory.identifier,
        )
    )

    assert runner._observe_spawning() == []
    assert runner.monitor.pending == {}
    assert runner.active_spawn_snapshot is None
    runner.close()


def test_replay_history_is_hidden_until_transaction_commit(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nflat\nH 0 0 0\n")
    config = SimulationConfig(
        provider="custom", geometry=xyz, geometry_units="bohr", num_states=2,
        initial_state=0, time_step=0.05, simulation_time=0.1,
        electronic_method="custom", coupling_mode="nac", spawn_threshold=1e9,
        write_xyz=True, run_directory=tmp_path / "transaction",
    )
    runner = SimulationRunner(
        config, CallableProvider(flat_provider, capabilities=ProviderCapabilities(nacs=True))
    )
    result = runner.propagate()
    runner.writer.begin_replay("test-window", entry_step=1, frontier_step=2)
    runner.writer.write_step(result.state, config.num_states, {"quantum_substeps": 8})
    assert [step["step"] for step in RunDataset(result.history).steps()] == [0, 1]
    assert sorted(path.name for path in (result.run_directory / "geometries").glob("step-*")) == [
        "step-00000000", "step-00000001",
    ]
    with (result.run_directory / "readable/populations.csv").open(newline="") as stream:
        assert [int(row["step"]) for row in csv.DictReader(stream)] == [0, 1]
    with h5py.File(result.history) as handle:
        assert "00000002" in handle["replay/test-window/steps"]
        assert "00000002" not in handle["steps"]
        assert "00000002" in handle["replay/test-window/superseded_steps"]
        assert int(handle["steps"].attrs["committed_through"]) == 1
    runner.writer.commit_replay("test-window", 2)
    assert [step["step"] for step in RunDataset(result.history).steps()] == [0, 1, 2]
    assert sorted(path.name for path in (result.run_directory / "geometries").glob("step-*")) == [
        "step-00000000", "step-00000001", "step-00000002",
    ]
    with (result.run_directory / "readable/populations.csv").open(newline="") as stream:
        assert [int(row["step"]) for row in csv.DictReader(stream)] == [0, 1, 2]
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
