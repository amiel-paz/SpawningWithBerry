"""Run an independent PySCF point on the published initial 4TCE geometry."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pyscf

from aims_berry.analysis.pyspawn import HARTREE_TO_EV
from aims_berry.electronic import ElectronicProperties, ElectronicStructureRequest
from aims_berry.electronic.pyscf import PySCFProvider
from aims_berry.geometry import atomic_numbers


def main(
    history: str = "reference-output/sim.hdf5",
    output: str = "reference-output/pyscf_4tce_initial_result.json",
) -> None:
    with h5py.File(history, "r") as handle:
        group = handle["traj_00"]
        atoms = tuple(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in group.attrs["atoms"]
        )
        geometry = group["positions"][0].reshape(-1, 3)
        reference_gap = float(np.diff(group["energies"][0])[0])

    provider = PySCFProvider(
        basis="6-31g*",
        charge=0,
        spin=0,
        scf_method="rhf",
        active_electrons=2,
        active_orbitals=2,
        state_weights=(0.5, 0.5),
        options={
            "density_fit": True,
            "max_memory": 12000,
            "scf_conv_tol": 1.0e-9,
            "casscf_conv_tol": 1.0e-7,
            "casscf_max_cycle": 50,
            "spin_penalty": 0.5,
        },
    )
    request = ElectronicStructureRequest(
        atoms=atoms,
        atomic_numbers=atomic_numbers(atoms),
        geometry=geometry,
        states=(0, 1),
        properties=frozenset({ElectronicProperties.ENERGIES}),
        active_state=1,
        time=0.0,
    )
    start = time.monotonic()
    result = provider.evaluate(request)
    elapsed = time.monotonic() - start
    gap = float(result.energies[1] - result.energies[0])
    difference_ev = abs(gap - reference_gap) * HARTREE_TO_EV
    summary = {
        "absolute_gap_difference_ev": difference_ev,
        "basis": "6-31g*",
        "density_fitting": True,
        "elapsed_seconds": elapsed,
        "energies_hartree": result.energies.tolist(),
        "gap_ev": gap * HARTREE_TO_EV,
        "gap_hartree": gap,
        "method": "SA(2)-CASSCF(2e,2o)",
        "pyscf_version": pyscf.__version__,
        "reference_gap_ev_physical": reference_gap * HARTREE_TO_EV,
        "reference_gap_hartree": reference_gap,
        "relative_gap_difference": abs(gap - reference_gap) / reference_gap,
        "spin_squares": result.metadata["spin_squares"],
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(target.resolve())
    if difference_ev > 0.05 or max(abs(value) for value in summary["spin_squares"]) > 1.0e-6:
        raise RuntimeError("PySCF initial-point corroboration failed")
    print("initial 4TCE PySCF gap corroborates the published raw history")


if __name__ == "__main__":
    main(*sys.argv[1:3])
