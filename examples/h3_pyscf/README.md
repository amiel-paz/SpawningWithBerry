# H3 PySCF benchmark

`aims.in` is a fast SA(2)-CASSCF(3e,3o)/STO-3G doublet calculation with analytic
PySCF gradients and NACs. It deliberately starts near the H3 conical-intersection
seam so a short run exercises spawning, HDF5 history, and exact restart.

Run `python run_ensemble.py` from this directory to execute seeds 1234, 2718, 3141,
and 5772 and export individual/mean population data. Each seed also creates bond,
angle, gap, and population products. Increase `simulation_time` for scientific
traces; the checked-in value is kept short for backend verification.
