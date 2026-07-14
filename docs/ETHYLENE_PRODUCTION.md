# Superseded SA(2) ethylene diagnostic

> This report describes the earlier SA(2)-CASSCF(2e,2o) diagnostic. It is not the
> requested production model. Production now uses equal-weight
> SA(3)-CASSCF(2e,2o), starts on S1, and must begin again from time zero.
> The current corroboration targets are defined in `ETHYLENE_REFERENCE.md`.

The production trajectory uses seed 87062, SA(2)-CASSCF(2e,2o)/6-31G, a 5 au
outer nuclear step, and adaptive metric-aware coefficient propagation down to
0.00244140625 au. PySCF evaluations run in three isolated worker processes with
two OpenMP threads each. Process isolation is required because concurrent PySCF
calls in threads do not preserve deterministic gradients and NACs.

## Staged validation

- The 700 au dt=10 gate accepted spawns at 460 and 600 au and completed with
  three TBFs. Its largest raw endpoint norm drift was 7.59e-11, minimum gap was
  0.00199 Eh, quantum-energy drift was 0.00237 Eh, and the largest per-TBF
  classical-energy drift was 4.73e-4 Eh.
- The first spawn rejected the child's redundant back-spawn at threshold entry;
  the second same-state spawn completed with 1024 coefficient-only substeps.
- Independent dt=5 and dt=10 trajectories located both coupling maxima at 460
  and 600 au. Threshold entry times differed by only 5 au.
- dt=10 was rejected: its state population differed from dt=5 by 0.0779 at
  600 au and coefficient convergence missed the 1e-6 gate at 950 au even after
  4096 substeps.
- dt=5 completed 1000 au. Its largest raw norm drift was 9.27e-11, minimum gap
  was 0.00197 Eh, quantum-energy drift was 0.00261 Eh, and populations at
  1000 au were S0=0.976677 and S1=0.023323.

Those initial staged runs exposed a later SA-CASSCF orbital-minimum switch. The
provider now performs finite multi-start orbital following while retaining the
declared 2e2o active space: it solves from the transported active pair and from an
active/external exchange, then selects the converged stationary solution with the
largest active-subspace overlap. It records candidate energies, active-subspace
singular values, CI-root overlaps, root assignment, and energy/gradient consistency
residuals on every call. A one-step sequential/process-isolated comparison matched
all positions, momenta, energies, gradients, NACs, amplitudes, matrices, events,
queue state, and RNG state exactly.

## Production status

The final consistent 2e2o run is valid through 945 au. It reproduces accepted spawn
maxima at 460 and 600 au, rejects the redundant 460-au child back-spawn at threshold
entry, retains three TBFs, and reaches populations S0=0.97657 and S1=0.02343 by
800 au. Norm, population, overlap, Hermiticity, spawn-shell, and energy gates pass
through the last committed frame. At 945 au the populations are S0=0.976572 and
S1=0.023428. Across committed frames, the largest raw norm drift is 9.27e-11,
population-sum error is 9.53e-14, Hermiticity residual is zero to stored precision,
the minimum retained overlap eigenvalue is 0.99784, quantum-energy drift is
0.00261 Eh, and the largest per-TBF classical-energy drift is 1.01e-4 Eh. The
minimum state gap is 0.00197 Eh at 600 au. Valid active-subspace overlaps remain
above 0.997 before the rejected step.

The requested 5000-au calculation is **not complete**. Propagation from 945 to
950 au is rejected because the best converged 2e2o candidate has active-subspace
singular overlaps `[0.5446, 0.4564]`, below the documented 0.7 continuity gate.
Adding three external-orbital multi-start candidates does not improve that result.
This is a loss of active-space identity, not a CI-root phase/permutation problem.

SA-CASSCF(2e,3o) removes the discontinuity in the tested region, but it is retained
only as diagnostic evidence because it is not the same electronic Hamiltonian as
the requested and previously benchmarked 2e2o calculation. No tolerance was
loosened, no path-dependent energy correction was applied, and no 2e3o result is
reported as 2e2o production.

Valid-through-945-au numerical datasets and plots can be exported with
`aims-berry analyze examples/ethylene_pyscf/run-production --input
examples/ethylene_pyscf/production.in --output analysis`.
