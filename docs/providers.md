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
