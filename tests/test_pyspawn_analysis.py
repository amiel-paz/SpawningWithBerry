import h5py
import numpy as np

from aims_berry.analysis import PySpawnDataset
from aims_berry.analysis.pyspawn import (
    HARTREE_TO_EV,
    PYSPAWN_PAPER_HARTREE_TO_EV,
)
from aims_berry.config import AU_TIME_PER_FS, BOHR_PER_ANGSTROM


def _write_trajectory(handle, label, times, positions, energies):
    group = handle.create_group(f"traj_{label}")
    group.attrs["atoms"] = np.asarray([b"H"] * 12)
    group.create_dataset("time", data=np.asarray(times)[:, None])
    group.create_dataset("positions", data=np.asarray(positions).reshape(len(times), -1))
    group.create_dataset("energies", data=np.asarray(energies))


def test_pyspawn_adapter_units_coordinates_and_coherent_populations(tmp_path):
    path = tmp_path / "sim.hdf5"
    geometry = np.zeros((12, 3))
    geometry[11, 0] = 2.0 * BOHR_PER_ANGSTROM
    with h5py.File(path, "w") as handle:
        sim = handle.create_group("sim")
        sim.attrs["labels"] = np.asarray([b"00", b"00b0"])
        sim.attrs["istates"] = np.asarray([1, 0])
        sim.create_dataset("quantum_time", data=[[0.0], [AU_TIME_PER_FS]])
        sim.create_dataset("num_traj_qm", data=[[1], [2]])
        sim.create_dataset(
            "qm_amplitudes",
            data=np.asarray([[1.0, 0.0], [0.5, np.sqrt(0.75)]], dtype=complex),
        )
        sim.create_dataset(
            "S",
            data=np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]),
        )
        _write_trajectory(
            handle,
            "00",
            [0.0, AU_TIME_PER_FS],
            [geometry, geometry],
            [[-1.0, -0.9], [-1.0, -0.8]],
        )
        _write_trajectory(
            handle,
            "00b0",
            [AU_TIME_PER_FS],
            [geometry],
            [[-1.0, -0.8]],
        )

    dataset = PySpawnDataset(path)
    assert np.allclose(dataset.coordinate("bond", (3, 11))["00"][:, 1], 2.0)
    assert np.allclose(dataset.energy_gap()[:, 1], [0.1, 0.2] * np.asarray(HARTREE_TO_EV))
    assert np.allclose(
        dataset.energy_gap(convention="pyspawn-paper")[:, 1],
        [0.1, 0.2] * np.asarray(PYSPAWN_PAPER_HARTREE_TO_EV),
    )
    populations = dataset.populations()
    assert np.allclose(populations[:, 0], [0.0, 1.0])
    assert np.allclose(populations[:, 1:], [[0.0, 1.0], [0.75, 0.25]])
