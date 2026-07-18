"""PySpawn's bundled cone test embedded in the same 3D space as aims_berry."""

import os

import numpy as np
import pyspawn
import pyspawn.general

clas_prop = "vv"
qm_prop = os.environ.get("PYSPAWN_QM_PROP", "fulldiag")
qm_ham = "adiabatic"
potential = "test_cone"
t0 = 0.0
ts = 0.1
tfinal = 200.0
numdims = 3
numstates = 2
traj_params = {
    "time": t0,
    "timestep": ts,
    "maxtime": tfinal,
    "spawnthresh": (0.5 * np.pi) / ts / 20.0,
    "istate": 1,
    "widths": np.asarray([6.0, 6.0, 6.0]),
    "masses": np.asarray([1822.0, 1822.0, 1.0e30]),
    "positions": np.asarray([0.45, 0.1, 0.0]),
    "momenta": np.asarray([-5.0, 0.0, 0.0]),
}
sim_params = {
    "quantum_time": t0,
    "timestep": ts,
    "max_quantum_time": tfinal,
    "qm_amplitudes": np.ones(1, dtype=np.complex128),
    "qm_energy_shift": -5.18,
}

exec("pyspawn.import_methods.into_simulation(pyspawn.qm_integrator." + qm_prop + ")")
exec("pyspawn.import_methods.into_simulation(pyspawn.qm_hamiltonian." + qm_ham + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.potential." + potential + ")")
exec("pyspawn.import_methods.into_traj(pyspawn.classical_integrator." + clas_prop + ")")
pyspawn.general.check_files()
trajectory = pyspawn.traj(numdims, numstates)
trajectory.set_parameters(traj_params)
simulation = pyspawn.simulation()
simulation.add_traj(trajectory)
simulation.set_parameters(sim_params)
simulation.propagate()
