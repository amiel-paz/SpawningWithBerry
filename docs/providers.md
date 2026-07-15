# Electronic-structure providers

Providers implement `ElectronicStructureProvider.evaluate(request)` and declare a
`ProviderCapabilities` object. Requests contain atoms, atomic numbers, geometry,
requested state indices/properties, the active state, time, and an optional previous
wavefunction state. Results contain state energies and may contain gradients, complex
NACs, state overlaps, dipoles, charges, a restart/guess handle, and metadata.

All arrays are validated before dynamics: energies `(nstate,)`, gradients
`(nstate,natom,3)`, NACs `(nstate,nstate,natom,3)`, and overlaps `(nstate,nstate)`.
Values must be finite and NACs must be anti-Hermitian in their state indices.

For a home-built calculator, wrap one callable without modifying the driver:

```python
from aims_berry import CallableProvider, ProviderCapabilities

provider = CallableProvider(
    calculator,
    capabilities=ProviderCapabilities(energies=True, gradients=True, nacs=True),
)
result = run(config, provider=provider)
```

`dump_state(directory)` and `load_state(directory, metadata)` let sophisticated
providers persist orbital/CI guesses or external checkpoint artifacts atomically with
the simulation. Subclassing `BaseProvider` supplies no-op implementations.

For large PySCF exploratory calculations, `provider_option density_fit true` enables
PySCF density fitting; `provider_option density_fit_auxbasis NAME` optionally selects
the auxiliary basis. It is opt-in because it changes the electronic approximation.

SA-CASSCF roots are spin-purified by default to the multiplicity implied by `spin`.
The result metadata records every root's measured `spin_squares`. Advanced workflows
may set `provider_option spin_square VALUE`, `spin_penalty VALUE`, or disable the
constraint with `provider_option fix_spin false`.

PySCF adiabatic roots always remain in energy order. Phase/root continuity is
measured with the many-electron determinant overlap transformed by the old/new
active-orbital overlap; raw CI coefficient dot products are not valid across
active-active orbital rotations. Before checkpointing, the provider applies the
unitary polar/Procrustes rotation that maximally aligns the new active orbitals to
the previous active orbitals and contragrediently transforms every CAS CI vector.
It then phase-aligns each energy-ordered many-electron root. This persistent gauge
transport changes only the wavefunction representation, not the physical CASSCF
state. `provider_option ci_root_overlap_min VALUE` sets the minimum same-root
overlap. Metadata contains `tracking_overlap` (physical orbital-aware overlap),
`active_orbital_rotation`, `aligned_active_overlap`, and both the pre-alignment
`ci_coefficient_overlap` and post-alignment `aligned_ci_coefficient_overlap`.
The Hungarian `root_assignment_suggestion` is diagnostic and never permutes
nondegenerate adiabatic energies.
