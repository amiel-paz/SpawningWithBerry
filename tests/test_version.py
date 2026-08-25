"""Version reporting and persisted provenance tests."""

from __future__ import annotations

import json

import h5py

from aims_berry import __version__, version_info
from aims_berry.cli import main
from aims_berry.config import SimulationConfig
from aims_berry.io.storage import HDF5Writer


def test_version_info_has_release_and_runtime_provenance() -> None:
    info = version_info()

    assert __version__ == "0.3.0"
    assert info["name"] == "aims-berry"
    assert info["version"] == __version__
    assert info["release_date"] == "2026-08-25"
    assert info["last_update"]
    assert info["python"]
    assert info["package_path"].endswith("aims_berry")


def test_version_command_supports_machine_readable_output(capsys) -> None:
    assert main(["version", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["version"] == __version__
    assert report["last_update"]


def test_history_records_source_provenance(tmp_path) -> None:
    geometry = tmp_path / "h.xyz"
    geometry.write_text("1\nH\nH 0 0 0\n")
    config = SimulationConfig(
        provider="callable",
        geometry=geometry,
        electronic_method="custom",
        simulation_time=1.0,
        run_directory=tmp_path / "run",
    )
    history = tmp_path / "simulation.h5"

    HDF5Writer(history, config)

    with h5py.File(history, "r") as handle:
        assert handle.attrs["aims_berry_version"] == __version__
        assert handle.attrs["aims_berry_last_update"]
        assert handle.attrs["aims_berry_source"] in {"git", "package"}
