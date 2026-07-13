# Prototype migration map

The former root-level research scripts are preserved unchanged in
`legacy/prototype`; generated trajectories, plots, movies, and text outputs are in
`legacy/results`. There are deliberately no compatibility shims.

| Legacy concept | New interface |
| --- | --- |
| `Bundle` and cross-linked global state | `SimulationState` |
| `Trajectory` | validated `TrajectoryBasisFunction` |
| Gaussian helper scripts | `aims_berry.dynamics.gaussian` |
| propagation scripts | `dynamics.quantum`, `dynamics.classical`, `SimulationRunner` |
| spawn/prune scripts | `aims_berry.spawning` strategies and monitors |
| model globals | `BerryModel2DParallelTransport` provider |
| ad hoc electronic calls | `ElectronicStructureProvider` request/result contract |
| output text trees | append-only `simulation.h5` plus atomic checkpoints |
| plotting scripts | `RunDataset` and `aims-berry analyze` |

The mathematical kernels are now import-safe and have no wildcard imports or
top-level executions. Generic electronic packages use the saddle-point Hamiltonian;
the analytic Berry provider additionally supplies connection and curvature hooks to
the classical integrator.
