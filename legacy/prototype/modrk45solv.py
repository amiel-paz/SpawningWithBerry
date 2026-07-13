from copy import deepcopy
import numpy as np
from scipy.integrate import solve_ivp

def pack_state(bundle):
    # 1) pack R, P for all trajectories
    R_list = []
    P_list = []
    for traj in bundle.trajectorylist:
        R_list.append(traj.rvec)
        P_list.append(traj.pvec)
    R = np.concatenate(R_list)
    P = np.concatenate(P_list)

    # 2) pack electronic coefficients
    C = bundle.GetC()  # complex (nC,)
    y = np.concatenate([R, P, C.real, C.imag])
    return y


def unpack_state(bundle, y):
    ntraj = bundle.ntraj
    ndim  = bundle.ndim
    nR    = ntraj * ndim
    nP    = ntraj * ndim
    nC    = len(bundle.GetC()) 

    R_flat   = y[0:nR]
    P_flat   = y[nR:nR+nP]
    Cre_flat = y[nR+nP : nR+nP + nC]
    Cim_flat = y[nR+nP + nC : nR+nP + 2*nC]

    # reshape / assign back into bundle
    for i, traj in enumerate(bundle.trajectorylist):
        traj.set_rvec(R_flat[i*ndim:(i+1)*ndim].copy())
        traj.set_pvec(P_flat[i*ndim:(i+1)*ndim].copy())

    C = Cre_flat + 1j*Cim_flat
    bundle.SetC(C)  # you’ll need a small setter like this

def renormalize_coeffs_in_bundle(bundle):
    """
    Renormalize electronic coefficients so that C† S C = 1
    for the *current* bundle state.
    """
    bundle.BuildHeff()
    S = bundle.S
    C = bundle.GetC()

    norm = np.vdot(C, S @ C)
    print(f"Norm before renormalization: {norm}")
    if norm == 0:
        raise RuntimeError("Zero S-norm: cannot renormalize C.")

    C /= np.sqrt(norm)

    # You will need to adapt this to your Bundle API:
    # either bundle.SetC(C) or direct attribute assignment.
    if hasattr(bundle, "SetC"):
        bundle.SetC(C)
    else:
        # Fallback: guess a common attribute name
        if hasattr(bundle, "C"):
            bundle.C[:] = C
        else:
            raise AttributeError("Bundle has no SetC or C attribute; please wire renormalization manually.")


def compute_quantum_energy(bundle):
    """
    Compute ⟨H⟩ = (C† H C) / (C† S C) for the current bundle.
    """
    bundle.BuildHeff()
    H = bundle.H
    S = bundle.S
    C = bundle.GetC()

    num = np.vdot(C, H @ C)
    denom = np.vdot(C, S @ C)
    if denom == 0:
        raise RuntimeError("Zero S-norm in compute_quantum_energy.")

    return (num / denom).real

def electronic_rhs(bundle):
    bundle.BuildHeff()
    S    = bundle.S
    H    = bundle.H
    Sdot = bundle.SDot
    C    = bundle.GetC()

    # PT gauge expectation
    H_expec = np.vdot(C, H @ C).real
    K   = (H - S*H_expec) - 1j*Sdot    # your Heff - i*tau

    # i S Cdot = K_eff C  -->  S Cdot = -1j K_eff C
    A0 = np.linalg.solve(S, K)          # shape (n,n)
    A_S = 0.5 * (A0 - np.linalg.solve(S, A0.conj().T @ S))

    # Back to K-space:
    K_S = S @ A_S
    rhs_vec = -1j * (K_S @ C)

    # Solve S Cdot = rhs_vec
    # For small basis, direct solve is fine:
    #Cdot = np.linalg.solve(S, rhs_vec)
    Cdot = np.linalg.solve(S, -1j * K @ C)

    # Or with CG:
    # Cdot, info = cg(S, rhs_vec, x0=None, tol=1e-10, maxiter=...)
    # if info != 0: handle / warn

    return Cdot

def classical_rhs(bundle):
    masses = np.asarray(bundle.trajectorylist[0].get_masses())
    G = np.diag(1.0 / masses)
    Rdot_list = []
    Pdot_list = []
    for traj in bundle.trajectorylist:
        state = traj.state  # however you store it
        Rdot = bundle.model.get_rdot(G, traj.rvec, traj.pvec, state)
        Pdot = bundle.model.get_pdot(G, traj.rvec, traj.pvec, state)
        Rdot_list.append(Rdot)
        Pdot_list.append(Pdot)

    Rdot_flat = np.concatenate(Rdot_list)
    Pdot_flat = np.concatenate(Pdot_list)

    return Rdot_flat, Pdot_flat

def rhs_full(t, y, bundle):
    # 1) unpack state into bundle (R, P, C)
    unpack_state(bundle, y)

    # 2) nuclear RHS
    Rdot_flat, Pdot_flat = classical_rhs(bundle)

    # 3) electronic RHS
    Cdot = electronic_rhs(bundle)  # complex (nC,)

    # 4) pack derivative: [Rdot, Pdot, Re(Cdot), Im(Cdot)]
    dydt = np.concatenate([Rdot_flat,
                           Pdot_flat,
                           Cdot.real,
                           Cdot.imag])
    return dydt


def compute_S_norm(bundle):
    """Return C^† S C (real scalar)."""
    bundle.BuildHeff()
    S = bundle.S
    C = bundle.GetC()
    return np.vdot(C, S @ C).real

def compute_classical_energies(bundle):
    """Return array of classical energies, one per trajectory."""
    Es = []
    masses = np.asarray(bundle.trajectorylist[0].get_masses())
    for traj in bundle.trajectorylist:
        state = traj.state  # or however you store it
        E = bundle.model.classical_energy(traj.rvec, traj.pvec, masses, state)
        Es.append(E)
    return np.array(Es, dtype=float)

def solverk45_and_step(
    bundle,
    dt,
    rtol=1.0e-6,
    atol=1.0e-9,
    norm_tol=1.0e-9,      # relative tolerance on S-norm drift
    eclass_tol=1.0e-9,    # absolute tolerance on classical E drift
    H_tol=1.0e-9,         # absolute tolerance on ⟨H⟩_end convergence
    max_attempts=6,
    verbose=True,
):
    """
    Advance (R, P, C) in `bundle` forward by time dt using RK45, with:

      - Outer adaptive sub-stepping: dt, dt/2, dt/4, ...
        (n_segments = 1, 2, 4, ..., 2^(max_attempts-1))

      - Per-segment S-norm renormalization of C
        (C ← C / sqrt(C† S C) after each RK45 microstep)

      - Diagnostics / tolerances:
          * S-norm drift between initial and final:
                norm_rel_drift = (⟨Ψ|Ψ⟩_final - ⟨Ψ|Ψ⟩_initial) / ⟨Ψ|Ψ⟩_initial
          * Classical energy drift per TBF between initial and final
          * Quantum energy ⟨H⟩_end convergence between adjacent attempts

      - Acceptance rule:
          * |norm_rel_drift|       <= norm_tol
          * worst |ΔE_classical|   <= eclass_tol
          * |⟨H⟩_end^(a) - ⟨H⟩_end^(a-1)| <= H_tol   (for a ≥ 2)

      The first attempt that satisfies all three is accepted.
      If none satisfy, the last attempt is committed with a warning.

    Returns (sol_last_segment, metrics_dict_for_accepted_or_last_step).
    """

    # Snapshot initial bundle state
    init_bundle = deepcopy(bundle)

    # Initial metrics (reference for *all* attempts)
    norm0 = compute_S_norm(init_bundle)
    Eclass0 = compute_classical_energies(init_bundle)
    H0 = compute_quantum_energy(init_bundle)

    if verbose:
        print("\n[RK45 adaptive] Initial state metrics:")
        print(f"  ⟨H⟩_initial   = {H0:.12f}")
        print(f"  S-norm_initial = {norm0:.12f}")
        print("  Classical E_initial (per TBF):")
        for i, Ei in enumerate(Eclass0):
            print(f"    traj {i:3d}: {Ei:.8f}")
        print()

    best_sol = None
    best_metrics = None
    prev_E_end = None  # ⟨H⟩ at end of previous attempt

    for attempt in range(max_attempts):
        nseg = 2**attempt                 # 1, 2, 4, 8, ...
        dt_seg = dt / nseg

        if verbose:
            print(f"\n[RK45 adaptive] Attempt {attempt+1}/{max_attempts} "
                  f"with {nseg} segment(s), dt_seg = {dt_seg:g}")

        # Work on a fresh copy of the initial state
        work_bundle = deepcopy(init_bundle)
        sol_last = None

        # Pack initial state for this attempt
        t0 = 0.0
        y = pack_state(work_bundle)

        # March over nseg segments of length dt_seg
        for seg in range(nseg):
            t_start = t0
            t_end   = t0 + dt_seg

            def f(t, y_vec, wb=work_bundle):
                return rhs_full(t, y_vec, wb)

            sol = solve_ivp(
                f,
                (t_start, t_end),
                y,
                method="RK45",
                rtol=rtol,
                atol=atol,
                t_eval=[t_end],  # only need end of this segment
            )
            if not sol.success:
                raise RuntimeError(
                    f"solve_ivp failed in attempt {attempt+1}, "
                    f"segment {seg+1}/{nseg}: {sol.message}"
                )

            # Update working bundle to end-of-segment state
            y_final = sol.y[:, -1]
            unpack_state(work_bundle, y_final)

            # Per-segment renormalization in S-metric
            renormalize_coeffs_in_bundle(work_bundle)

            # Repack for next segment
            y = pack_state(work_bundle)
            t0 = t_end
            sol_last = sol  # keep last segment solution

        # ---- Metrics for this *full dt* attempt ----
        norm1 = compute_S_norm(work_bundle)
        Eclass1 = compute_classical_energies(work_bundle)
        H_end = compute_quantum_energy(work_bundle)

        norm_drift = norm1 - norm0
        norm_rel_drift = norm_drift / (norm0 + 1e-300)

        dE_class = Eclass1 - Eclass0
        worst_dE = float(np.max(np.abs(dE_class)))
        worst_dE_idx = int(np.argmax(np.abs(dE_class)))

        metrics = {
            "attempt": attempt + 1,
            "n_segments": nseg,
            "dt_segment": dt_seg,
            "norm_initial": norm0,
            "norm_final": norm1,
            "norm_abs_drift": norm_drift,
            "norm_rel_drift": norm_rel_drift,
            "Eclass_initial": Eclass0,
            "Eclass_final": Eclass1,
            "Eclass_dE": dE_class,
            "Eclass_worst_dE": worst_dE,
            "Eclass_worst_traj_idx": worst_dE_idx,
            "H_initial": H0,
            "H_end": H_end,
        }

        if prev_E_end is not None:
            dH = abs(H_end - prev_E_end)
            metrics["dH_vs_prev"] = dH
        else:
            dH = None

        # ---- Verbose reporting ----
        if verbose:
            print(f"  ⟨H⟩: {H0:.12f} → {H_end:.12f}", end="")
            if dH is not None:
                print(f"   (|Δ⟨H⟩_end| vs prev attempt = {dH:.3e})")
            else:
                print("   (first attempt, no previous ⟨H⟩ to compare)")

            print(f"  S-norm: {norm0:.12f} → {norm1:.12f} "
                  f"(Δ = {norm_drift:.3e}, rel Δ = {norm_rel_drift:.3e})")

            print("  Classical E (per TBF):")
            for i, (Ei0, Ei1, dEi) in enumerate(zip(Eclass0, Eclass1, dE_class)):
                print(f"    traj {i:3d}: {Ei0:.8f} → {Ei1:.8f} (Δ = {dEi:.3e})")
            print(f"  Worst |ΔE_class| = {worst_dE:.3e} on traj {worst_dE_idx}")

            print(f"  Tolerances: "
                  f"norm_tol={norm_tol:.1e}, "
                  f"eclass_tol={eclass_tol:.1e}, "
                  f"H_tol={H_tol:.1e}")

        # ---- Checks ----
        norm_ok   = (abs(norm_rel_drift) <= norm_tol)
        eclass_ok = (worst_dE <= eclass_tol)
        # H convergence only meaningful from attempt 2 onwards
        if dH is None:
            H_ok = False   # force at least one refinement before using H_tol
        else:
            H_ok = (dH <= H_tol)

        if verbose:
            print(f"  norm_ok={norm_ok}, eclass_ok={eclass_ok}, H_ok={H_ok}")

        best_sol = sol_last
        best_metrics = metrics
        prev_E_end = H_end

        # Acceptance condition:
        #   - norm and classical energy within tolerance
        #   - and ⟨H⟩_end converged vs previous attempt
        if norm_ok and eclass_ok and H_ok:
            if verbose:
                print(f"  ✅ Accepted attempt {attempt+1} "
                      f"with {nseg} segment(s).")
            # Accept this attempt: commit work_bundle → bundle
            bundle.__dict__.update(work_bundle.__dict__)
            best_metrics["accepted"] = True
            return best_sol, best_metrics

        # otherwise, go to next attempt with finer segmentation

    # If we exit the loop without break: no attempt met all tolerances
    if verbose:
        print("\n⚠️  WARNING: Reached max_attempts without satisfying "
              "all tolerances (norm, classical E, ⟨H⟩ convergence).")
        print("    Committing last attempt's state anyway; "
              "inspect metrics carefully.")

    # Commit last attempt state
    bundle.__dict__.update(work_bundle.__dict__)
    if best_metrics is None:
        best_metrics = {}
    best_metrics["accepted"] = False
    return best_sol, best_metrics

def solverk45_and_step_classical(
    bundle,
    dt,
    rtol=1.0e-6,
    atol=1.0e-9,
    eclass_tol=1.0e-9,    # absolute tolerance on classical E drift
    max_attempts=6,
    verbose=True,
):
    """
    Advance (R, P) in `bundle` forward by time dt using RK45, with:

      - Outer adaptive sub-stepping: dt, dt/2, dt/4, ...
        (n_segments = 1, 2, 4, ..., 2^(max_attempts-1))

      - Diagnostics / tolerances:
          * Classical energy drift per TBF between initial and final

      - Acceptance rule:
          * worst |ΔE_classical|   <= eclass_tol

      The first attempt that satisfies all three is accepted.
      If none satisfy, the last attempt is committed with a warning.

    Returns (sol_last_segment, metrics_dict_for_accepted_or_last_step).
    """

    # Snapshot initial bundle state
    init_bundle = deepcopy(bundle)

    # Initial metrics (reference for *all* attempts)
    Eclass0 = compute_classical_energies(init_bundle)

    if verbose:
        print("\n[RK45 adaptive] Initial state metrics:")
        print("  Classical E_initial (per TBF):")
        for i, Ei in enumerate(Eclass0):
            print(f"    traj {i:3d}: {Ei:.8f}")
        print()

    best_sol = None
    best_metrics = None
    prev_E_end = None  # ⟨H⟩ at end of previous attempt

    for attempt in range(max_attempts):
        nseg = 2**attempt                 # 1, 2, 4, 8, ...
        dt_seg = dt / nseg

        if verbose:
            print(f"\n[RK45 adaptive] Attempt {attempt+1}/{max_attempts} "
                  f"with {nseg} segment(s), dt_seg = {dt_seg:g}")

        # Work on a fresh copy of the initial state
        work_bundle = deepcopy(init_bundle)
        sol_last = None

        # Pack initial state for this attempt
        t0 = 0.0
        y = pack_state(work_bundle)

        # March over nseg segments of length dt_seg
        for seg in range(nseg):
            t_start = t0
            t_end   = t0 + dt_seg

            def f(t, y_vec, wb=work_bundle):
                unpack_state(wb, y_vec)
                Rdot_flat, Pdot_flat = classical_rhs(wb)
                Cbuffer = np.zeros(len(wb.C.real))
                return np.concatenate([Rdot_flat,Pdot_flat,Cbuffer,Cbuffer])

            sol = solve_ivp(
                f,
                (t_start, t_end),
                y,
                method="RK45",
                rtol=rtol,
                atol=atol,
                t_eval=[t_end],  # only need end of this segment
            )
            if not sol.success:
                raise RuntimeError(
                    f"solve_ivp failed in attempt {attempt+1}, "
                    f"segment {seg+1}/{nseg}: {sol.message}"
                )

            # Update working bundle to end-of-segment state
            y_final = sol.y[:, -1]
            unpack_state(work_bundle, y_final)

            # Repack for next segment
            y = pack_state(work_bundle)
            t0 = t_end
            sol_last = sol  # keep last segment solution

        # ---- Metrics for this *full dt* attempt ----
        Eclass1 = compute_classical_energies(work_bundle)

        dE_class = Eclass1 - Eclass0
        worst_dE = float(np.max(np.abs(dE_class)))
        worst_dE_idx = int(np.argmax(np.abs(dE_class)))

        metrics = {
            "attempt": attempt + 1,
            "n_segments": nseg,
            "dt_segment": dt_seg,
            "Eclass_initial": Eclass0,
            "Eclass_final": Eclass1,
            "Eclass_dE": dE_class,
            "Eclass_worst_dE": worst_dE,
            "Eclass_worst_traj_idx": worst_dE_idx,
        }

        # ---- Verbose reporting ----
        if verbose:

            print("  Classical E (per TBF):")
            for i, (Ei0, Ei1, dEi) in enumerate(zip(Eclass0, Eclass1, dE_class)):
                print(f"    traj {i:3d}: {Ei0:.8f} → {Ei1:.8f} (Δ = {dEi:.3e})")
            print(f"  Worst |ΔE_class| = {worst_dE:.3e} on traj {worst_dE_idx}")

            print(f"  Tolerances: "
                  f"eclass_tol={eclass_tol:.1e}")

        # ---- Checks ----
        eclass_ok = (worst_dE <= eclass_tol)

        if verbose:
            print(f"  eclass_ok={eclass_ok}")

        best_sol = sol_last
        best_metrics = metrics

        # Acceptance condition:
        #   - norm and classical energy within tolerance
        #   - and ⟨H⟩_end converged vs previous attempt
        if eclass_ok:
            if verbose:
                print(f"  ✅ Accepted attempt {attempt+1} "
                      f"with {nseg} segment(s).")
            # Accept this attempt: commit work_bundle → bundle
            bundle.__dict__.update(work_bundle.__dict__)
            best_metrics["accepted"] = True
            return best_sol, best_metrics

        # otherwise, go to next attempt with finer segmentation

    # If we exit the loop without break: no attempt met all tolerances
    if verbose:
        print("\n⚠️  WARNING: Reached max_attempts without satisfying "
              "classical E tolerance.")
        print("    Committing last attempt's state anyway; "
              "inspect metrics carefully.")

    # Commit last attempt state
    bundle.__dict__.update(work_bundle.__dict__)
    if best_metrics is None:
        best_metrics = {}
    best_metrics["accepted"] = False
    return best_sol, best_metrics

