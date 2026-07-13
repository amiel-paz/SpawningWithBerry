# Ethylene integration example

This SA(2)-CASSCF(2e,2o)/6-31G closed-shell example exercises a larger PySCF
backend, the C-C and H-C-C-H observables, state gap/population output, and restart.
It is intentionally a slower integration workload rather than part of the fast test
suite.

`production.in` is the 5000 au (120.94 fs), 500-step production analogue of the
original PySpawn ethylene example. It uses that example's equilibrium geometry,
Hessian, random seed, timestep, final time, active space, basis, and spawning
threshold, with spin-pure density-fitted PySCF replacing TeraChem. Its restartable
output is written to the ignored `run-production/` directory.
