import numpy as np
from bundle import *
from multidgaussfuncs import *

def sample_phase_space_energy_MC(traj, model, nsamples, rng=None):
    """
    Monte Carlo estimate of the quantum expectation value of H
    using phase-space sampling of the nuclear Gaussian's Wigner-like blob.

    We draw (r,p) samples from the minimum-uncertainty Gaussian
    implied by the trajectory's Heller packet, evaluate the classical
    energy functional on each draw, and average.

    Parameters
    ----------
    model : BerryModel2DParallelTransport (your model object)
    traj  : Trajectory
        Needs fields:
          - traj.rvec    (ndim,)
          - traj.pvec    (ndim,)  canonical centroid momentum
          - traj.widths  (ndim,)  'a' parameters of each 1D Gaussian
          - traj.masses  (ndim,)
          - traj.state   int, 0 or 1
    nsamples : int
        How many Monte Carlo samples to draw.
    rng : np.random.Generator or None
        Pass in a Generator for reproducibility.
        If None, we'll make a new default Generator.

    Returns
    -------
    E_mean : float
        Monte Carlo mean energy.
    E_std  : float
        Monte Carlo standard error of the mean (std/sqrt(N)).
    """

    if rng is None:
        rng = np.random.default_rng()

    r0   = np.asarray(traj.rvec,    dtype=float)  # centroid position
    p0   = np.asarray(traj.pvec,    dtype=float)  # canonical momentum at centroid
    avec = np.asarray(traj.widths,  dtype=float)  # Gaussian width 'a' per dim
    mvec = np.asarray(traj.masses,  dtype=float)  # masses
    st   = traj.state

    ndim = r0.shape[0]

    # Position and momentum standard deviations per DOF:
    # For chi(R) ~ exp[-a (R-R0)^2 + i P0 (R-R0)], with ħ=1:
    #   Var(R) = 1/(4a),    so sigma_R = 1/(2 sqrt(a))
    #   Var(P) = a,         so sigma_P = sqrt(a)
    sigma_R = 1.0 / (2.0 * np.sqrt(avec))
    sigma_P = np.sqrt(avec)

    energies = np.zeros(nsamples, dtype=float)

    for s in range(nsamples):
        # sample each dimension independently
        dR = rng.normal(loc=0.0, scale=sigma_R, size=ndim)
        dP = rng.normal(loc=0.0, scale=sigma_P, size=ndim)

        r_samp = r0 + dR       # sampled position
        p_samp = p0 + dP       # sampled canonical momentum

        # evaluate the classical energy functional on this sampled phase point
        E_s = model.classical_energy(r_samp, p_samp, mvec, st)
        energies[s] = E_s

    E_mean = np.mean(energies)
    E_sem  = np.std(energies, ddof=1) / np.sqrt(nsamples)  # standard error

    return E_mean, E_sem

def projected_refined_energy_MC(traj, model, nsamples, include_original=False, rng=None, Eclass_deviation=None):
    """
    Build a local multi-Gaussian bundle around `traj`,
    project the original TBF into that span,
    and compute its properly normalized energy expectation.

    Returns a *real* scalar up to numerical noise.
    """

    if rng is None:
        rng = np.random.default_rng()

    ndim   = traj.ndim
    bundle = Bundle(ndim, model)

    r0   = np.asarray(traj.rvec,    dtype=float)  # centroid R0
    p0   = np.asarray(traj.pvec,    dtype=float)  # centroid P0 (canonical)
    avec = np.asarray(traj.widths,  dtype=float)  # widths a_k
    mvec = np.asarray(traj.masses,  dtype=float)  # masses
    st   = traj.state

    Eclass_desired = model.classical_energy(r0,p0,mvec,st)

    # phase-space widths of the Wigner-ish blob
    sigma_R = 1.0 / (2.0 * np.sqrt(avec))  # sqrt(Var(R))
    sigma_P = np.sqrt(avec)                # sqrt(Var(P))

    # we'll build nsamples sampled TBFs + 1 central TBF
    overlaps = np.zeros(nsamples + int(include_original), dtype=np.complex128)

    # 1. sample local Gaussians in phase space, add them to bundle
    accepted = 0
    while (accepted < nsamples):
        dR = rng.normal(loc=0.0, scale=sigma_R, size=ndim)
        dP = rng.normal(loc=0.0, scale=sigma_P, size=ndim)

        r_samp = r0 + dR
        p_samp = p0 + dP

        Eclass_candidate = model.classical_energy(r_samp,p_samp,mvec,st)
        if (Eclass_deviation is not None):
           deviation = abs(1 - Eclass_candidate/Eclass_desired)
           if deviation > Eclass_deviation:
               continue
           else:
               print(f"Trajectory added with {100*deviation}% deviation from total energy.")

        bundle.add_trajectory(r_samp, p_samp, st)
        TBF_samp = bundle.trajectorylist[-1]

        # overlap < sample | central >
        overlaps[accepted] = ApplyToTraj(TBF_samp, traj, OverlapR, False)
        accepted += 1

    # 2. append the original central TBF itself as the last basis function
    if (include_original):
        bundle.add_trajectory(r0, p0, st)
        overlaps[-1] = 1.0 + 0.0j  # exactly <central|central>

    # 3. Build S and H in this local basis
    bundle.BuildS()       # should be analytic / essentially Hermitian
    bundle.BuildSp5inv()  # gives you bundle.Sinv
    bundle.BuildH()       # uses KE analytic + PE quadrature

    # 4. Enforce Hermiticity of BOTH S and H at this stage.
    #    This is CRITICAL. Do it *after* BuildH(), not inside.
    H_raw = bundle.H
    S_raw = bundle.S

    H_herm = 0.5 * (H_raw + H_raw.conjugate().T)
    S_herm = 0.5 * (S_raw + S_raw.conjugate().T)

    # 5. Recompute expansion coefficients C that best reproduce the central TBF
    #    inside this subspace.
    #    You previously used Sinv @ overlaps (which is fine as a linear solve),
    #    but now we'll solve against the Hermitized S to stay consistent.
    C = np.linalg.solve(S_herm, overlaps)

    # (Optionally stash C back into bundle if you want diagnostics)
    bundle.SetC(C)

    # 6. Generalized Rayleigh quotient:
    #    E = (C† H C)/(C† S C)
    num = C.conjugate().T @ (H_herm @ C)
    den = C.conjugate().T @ (S_herm @ C)
    E_proj = num / den
    # 7. For paranoia/debug, you can also compute how anti-Hermitian H was:
    #    ||H - H†||
    antiH_norm = np.linalg.norm(H_raw - H_raw.conjugate().T)
    # and how far from normalization you were:
    norm_before = (bundle.C.conjugate().T @ (S_raw @ bundle.C))

    # These could be logged if you like:
    # print("antiHermitian leakage in H:", antiH_norm)
    # print("C† S_raw C before Herm fix:", norm_before)

    return bundle, E_proj

