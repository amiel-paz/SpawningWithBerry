# Ethylene corroboration reference

The primary reference is H. Tao, B. G. Levine, and T. J. Martinez,
"Ab Initio Multiple Spawning Dynamics Using Multi-State Second-Order Perturbation
Theory," J. Phys. Chem. A 113, 13656-13662 (2009),
https://doi.org/10.1021/jp9063565.

Our direct comparison is the paper's CASSCF control, not its dynamically correlated
MS-CASPT2 calculation. Both use the same equal-weight, three-root zeroth-order model:
SA(3)-CAS(2e,2o)/6-31G*. Our PySCF provider supplies analytic SA-CASSCF gradients
and NACs in place of the paper's electronic-structure implementation.

## Electronic sanity gate

- Initial bright state: S1, with valence pi-pi* character.
- Reference vertical S0-to-S1 excitation at CASSCF: 10.12 eV.
- Corresponding MS-CASPT2 value: 8.62 eV; experiment: 7.66 eV.
- All three lowest singlet states are averaged with equal weights.

The 10.12 eV CASSCF gap is the first gate. A material discrepancy must be traced to
the geometry, orbital/state selection, density fitting, or implementation before a
production trajectory is interpreted.

Current status: at the inherited PySpawn equilibrium geometry (C-C = 1.320885
angstrom), PySCF gives 10.3580 eV. Turning density fitting off changes this to only
10.3592 eV, so density fitting is not the source of the 0.238 eV discrepancy. At the
separate 1.339 angstrom example geometry, PySCF gives 10.2328 eV. This geometry is
accepted for production with an explicitly documented 0.1128 eV deviation from the
paper; no energy threshold was relaxed to force agreement.

An MP2/6-31G* optimization starting from the 1.339 angstrom geometry converged in
four geometry steps to C-C = 1.336420 angstrom and C-H = 1.085028 angstrom. The
final MP2 energy is -78.2870058554 Eh, gradient RMS is 7.05e-7 Eh/bohr, and maximum
gradient is 2.00e-6 Eh/bohr. At this geometry the spin-pure SA(3)-CAS(2e,2o)/6-31G*
gap is 10.2477 eV with density fitting and 10.2489 eV without density fitting. Thus,
MP2 optimization reduces the discrepancy to 0.129 eV but does not by itself recover
10.12 eV. The diagnostic geometry is stored as
`examples/ethylene_pyscf/ethylene_mp2_631gstar.xyz`; it has not replaced the
production equilibrium geometry or its associated Hessian.

Production uses the accepted 1.339/1.086 angstrom, 117.6 degree geometry and a
numerical MP2/6-31G* Hessian evaluated there. The Hessian is symmetrized in Cartesian
coordinates and projected in mass-weighted coordinates to remove all three
translations and three rotations. Its remaining 12 eigenmodes are the vibrational
modes sampled by the Wigner initializer.

## Dynamics protocol and numerical targets

- Sample independent initial conditions from the vibrational ground-state Wigner
  distribution in the harmonic approximation.
- Propagate from S1 for 250 fs.
- The paper averages 13 initial conditions. Its 155 spawned TBFs correspond to about
  12 spawned functions per initial condition.
- More than 97% of the ensemble population reaches S0 by 250 fs.
- The CASSCF S1 lifetime from an exponential fit is 110 +/- 6 fs. The MS-CASPT2
  comparison is 89 +/- 3 fs.

A single trajectory may test stability and spawning behavior, but it cannot
corroborate the lifetime or branching fractions. Those require the full ensemble and
population averaging.

## Figures and observables to reproduce

1. State populations versus time, with an ensemble S1 lifetime near 110 fs for the
   CASSCF calculation (paper Figure 1).
2. Spawning geometries classified as twisted-pyramidalized or ethylidene-like (paper
   Figure 2).
3. Population transferred and transfer efficiency versus the minimum S1/S0 gap for
   each spawning event (paper Figure 3).
4. Nonbonded C-H distances for hydrogen migration and the carbon pyramidalization
   angle for representative TBFs (paper Figure 4).
5. H/H2 elimination coordinates after return to S0, as a secondary long-time
   observable (paper Figure 5).

At the CASSCF level, the paper reports that twisted-pyramidalized and ethylidene-like
intersections each account for roughly half of the transferred population. Dynamic
correlation changes this branching in favor of the twisted-pyramidalized path, so our
CASSCF result should be compared to the approximately even CASSCF split.

## Interpretation boundaries

Agreement means reproducing the CASSCF population time scale, controlled spawning,
the two decay geometries, and the gap-dependent transfer trends within ensemble
uncertainty. We do not expect the CASSCF calculation to reproduce the 8.62 eV
MS-CASPT2 excitation, its 89 fs lifetime, or its dynamically correlated branching
ratio.
