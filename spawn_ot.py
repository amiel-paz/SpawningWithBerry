import numpy as np
import copy
from bundle import Bundle
from classprop import *
from propagate import PropH
import cma   # pip install cma

def energy_matched_canonical_momentum(r_child, u_dir, model, mass, E_parent,
                                       parent_phi, target_state):
    """
    Given:
        r_child     : np.array([x,y])
        u_dir       : np.array([ux, uy]) *unit*
        model       : BerryModel2DParallelTransport
        mass        : array-like [m_x, m_y]
        E_parent    : scalar (classical) energy of parent
        parent_phi  : Φ_parent = V + D_off + divd at parent's location/state
        target_state: 0 or 1

    Try to find α so that P_child = α*u_dir has *the same* classical energy
    on the *target_state* surface.

    Returns:
        P_child (2,)  if feasible
        None          if impossible (negative discriminant or K_target <= 0)
    """
    r_child = np.asarray(r_child, dtype=float)
    u_dir   = np.asarray(u_dir, dtype=float)

    # A_child and Φ_child at this point/state
    A_c = model.A(r_child, target_state).real
    Phi_c = (
        model.V_adiabatic(r_child, target_state)
        + model.D_off(r_child, mass, target_state)
        + model.div_d(r_child, mass, target_state, target_state)
    ).real

    inv_m = 1.0 / np.asarray(mass, float)

    # target kinetic metric term
    # E = 1/2 (P+A)^T G (P+A) + Φ  ==>  (P+A)^T G (P+A) = 2 (E - Φ)
    K_target = 2.0 * (E_parent - Phi_c)
    if K_target <= 0.0:
        # cannot match energy at this geometry
        return None

    # build quadratic in α
    # (α u + A)^T G (α u + A) = K_target
    # a α^2 + b α + c = 0
    a = np.dot(u_dir * inv_m, u_dir)            # u^T G u
    b = 2.0 * np.dot(u_dir * inv_m, A_c)        # 2 u^T G A
    c = np.dot(A_c * inv_m, A_c) - K_target     # A^T G A - K_target

    disc = b*b - 4.0*a*c
    if disc < 0.0:
        return None

    sqrt_disc = np.sqrt(disc)

    # two possible α
    alpha1 = (-b + sqrt_disc) / (2.0 * a)
    alpha2 = (-b - sqrt_disc) / (2.0 * a)

    # heuristic: pick the one with larger |α| (often gives the "real" momentum)
    # you can also bias by sign if you want
    if abs(alpha1) >= abs(alpha2):
        alpha = alpha1
    else:
        alpha = alpha2

    P_child = alpha * u_dir
    return P_child


def make_child_like(parent_traj, r_child, p_child, state_child):
    """
    Make a minimal 'trajectory-like' object that has the same widths/masses
    as the parent but new r,p,state, so we can call model.eval_integrals(...)
    without mutating the real bundle.
    """
    # WARNING: this assumes your Trajectory class looks like bundle.trajectorylist[...]
    # and that we can shallow-copy it.
    child = copy.deepcopy(parent_traj)
    child.set_rvec(np.asarray(r_child, float))
    child.set_pvec(np.asarray(p_child, float))
    child.state = state_child
    return child


def make_spawn_objective(parent_traj, model, target_state):
    """
    Returns a callable f(x) for CMA-ES.
    x = [dx, dy, theta]
    where
        R_child = R_parent + [dx, dy]
        u_dir   = (cos theta, sin theta)
    We
      - energy-match canonical momentum
      - if success: build a child TBF and evaluate |H_pc|
      - CMA = minimize, so we return -|H_pc|
    """
    mass         = parent_traj.get_masses()
    R_parent     = parent_traj.get_rvec()
    P_parent     = parent_traj.get_pvec()
    parent_state = parent_traj.get_state()

    # parent energy
    E_parent = model.classical_energy(R_parent, P_parent, mass, parent_state)

    # parent Φ (will help us keep things symmetric, but we can recompute)
    Phi_parent = (
        model.V_adiabatic(R_parent, parent_state)
        + model.D_off(R_parent, mass, parent_state)
        + model.div_d(R_parent, mass, parent_state, parent_state)
    ).real

    def f(x):
        dx, dy, theta = x
        R_child = R_parent + np.array([dx, dy], float)
        u_dir = np.array([np.cos(theta), np.sin(theta)], float)

        # try to match energy
        P_child = energy_matched_canonical_momentum(
            R_child, u_dir, model, mass,
            E_parent, Phi_parent, target_state
        )
        if P_child is None:
            # hard penalty
            return 1.0e6

        # build a temporary child
        child_traj = make_child_like(parent_traj, R_child, P_child, target_state)

        # evaluate coupling
        H_pc = model.eval_integrals(parent_traj, child_traj, True) #Last bool governs whether to evaluate KE 

        # maximize |H_pc|  → minimize -|H_pc|
        return -abs(H_pc)

    return f

def make_spawn_objective_exact(C_parent, parent_traj, model, dt, target_state, nslice=20):
    """
    Returns a callable f(x) for CMA-ES.
    x = [dx, dy, theta]
    where
        R_child = R_parent + [dx, dy]
        u_dir   = (cos theta, sin theta)
    We
      - energy-match canonical momentum
      - if success: build a child TBF and evaluate TDSE to obtain C_child(t+dt), given C_child(t) = 0
      - CMA = minimize, so we return -C_child(t+dt)
    """
    mass         = parent_traj.get_masses()
    R_parent     = parent_traj.get_rvec()
    P_parent     = parent_traj.get_pvec()
    parent_state = parent_traj.get_state()

    # parent energy
    E_parent = model.classical_energy(R_parent, P_parent, mass, parent_state)

    # parent Φ (will help us keep things symmetric, but we can recompute)
    Phi_parent = (
        model.V_adiabatic(R_parent, parent_state)
        + model.D_off(R_parent, mass, parent_state)
        + model.div_d(R_parent, mass, parent_state, parent_state)
    ).real

    def f(x):
        dx, dy, theta = x
        R_child = R_parent + np.array([dx, dy], float)
        u_dir = np.array([np.cos(theta), np.sin(theta)], float)

        # try to match energy
        P_child = energy_matched_canonical_momentum(
            R_child, u_dir, model, mass,
            E_parent, Phi_parent, target_state
        )
        if P_child is None:
            # hard penalty
            return 1.0e6

        # build a temporary child
        child_traj = make_child_like(parent_traj, R_child, P_child, target_state)

        # propagate instantaneous Hamiltonian for dt
        temp_bundle = Bundle(model.ndim,model)
        temp_bundle.add_trajectory(R_parent,P_parent,parent_state) 
        temp_bundle.add_trajectory(R_child,P_child,target_state)
        temp_bundle.BuildHeff() 
        C_old = np.asarray([C_parent, 0])
        C_new = PropH(temp_bundle.Heff, C_old, dt, nslice, renormalize=False)
        nS = np.vdot(C_new, temp_bundle.S @ C_new).real
        C_new /= np.sqrt(nS)

        # maximize |C(t+dt)|  → minimize -|C(t+dt)|
        return -abs(C_new[1])

    return f

def optimize_spawn(parent_traj,
                   target_state,
                   model,
                   mass,
                   rng,
                   verbose=False):
    """
    Runs CMA-ES (or your 2D random search for now), returns (R_child, P_child, best_val, info)
    """
    # --- 1. get parent info
    R_p = parent_traj.get_rvec()
    P_p = parent_traj.get_pvec()
    s_p = parent_traj.get_state()

    # --- 2. propose an initial child (you can replace with actual CMA-ES)
    # sample position near parent:
    R0 = R_p + 0.05 * rng.normal(size=2)

    # sample unit direction in p
    u = rng.normal(size=2)
    u /= np.linalg.norm(u)

    # energy-match: E_class(parent) = E_class(child)
    E_parent = model.classical_energy(R_p, P_p, mass, s_p)

    # now pick magnitude for child momentum so its energy matches
    # here I’ll just try the parent’s magnitude first:
    P0 = np.linalg.norm(P_p) * u

    E_child_guess = model.classical_energy(R0, P0, mass, target_state)
    if verbose:
        print("[spawn-opt] parent E =", E_parent)
        print("[spawn-opt] child guess E =", E_child_guess)

    # if energy too off, mark as bad
    if abs(E_child_guess - E_parent) > 1.0e-3:
        best_val = -1.0e9
        if verbose:
            print("[spawn-opt] energy mismatch, rejecting")
        return R0, P0, best_val, {"reason": "energy mismatch"}

    # otherwise evaluate “coupling” objective at this guess
    d_pc = model.d(R0, s_p, target_state)
    coup = abs(np.dot(P_p, d_pc))  # your objective

    if verbose:
        print(f"[spawn-opt] candidate R={R0}, P={P0}, |P·d|={coup:.6e}")

    return R0, P0, coup, {}

def measure_self_spawn_coupling(parent_traj, model, target_state):
    """
    Build a *temporary* 2-TBF bundle:
      TBF0 = parent (as-is)
      TBF1 = same geometry/momenta but on target_state
    Then return |(H - i Sdot)_{01}|.
    """
    tmp_bundle = Bundle(model.ndim, model)

    # parent, as-is
    R_p = parent_traj.get_rvec().copy()
    P_p = parent_traj.get_pvec().copy()
    s_p = parent_traj.get_state()
    m_p = parent_traj.get_masses()

    tmp_bundle.add_trajectory(R_p, P_p, s_p)

    # clone on other surface
    s_c = target_state
    tmp_bundle.add_trajectory(R_p, P_p, s_c)

    # build matrices
    tmp_bundle.BuildHeff()

    K = tmp_bundle.H - 1j * tmp_bundle.SDot

    return abs(K[0, 1])

def should_spawn(bundle,
                 model,
                 step,
                 nstates=2,
                 thresh=1e-3,
                 cooldown_steps=200,
                 forbid_back_steps=100,
                 min_pop=1.0e-4,
                 verbose=False):
    """
    spawn_meta: dict[int -> {last_spawn_step, last_spawn_target}]
    """
    spawn_list = []
    ntraj = bundle.ntraj

    # need coeffs for pop threshold
    bundle.BuildS()
    C = bundle.GetC()
    S = bundle.S
    pops = np.real(np.conjugate(C) * (S @ C))  # crude per-slot contribution

    for i in range(ntraj):
        parent = bundle.trajectorylist[i]
        s_parent = parent.state

        # make sure meta exists
        if i not in bundle.spawn_meta:
            bundle.spawn_meta[i] = {"last_spawn_step": -10**9, "last_spawn_target": None}

        # cooldown check
        if step - bundle.spawn_meta[i]["last_spawn_step"] < cooldown_steps:
            continue

        # pop check
        if pops[i] < min_pop:
            continue

        for s_target in range(nstates):
            if s_target == s_parent:
                continue

            # anti-bounce: don’t spawn back to where you just spawned from
            if (bundle.spawn_meta[i]["last_spawn_target"] == s_target
                and step - bundle.spawn_meta[i]["last_spawn_step"] < forbid_back_steps):
                continue

            mag = measure_self_spawn_coupling(parent, model, s_target)

            if verbose:
                print(f"[spawn-check] step {step} traj {i}: {s_parent}->{s_target}, |K|={mag:.3e}")

            if mag > thresh:
                spawn_list.append((i, s_target))

    return spawn_list


def backprop_child(child_traj, model, dt, N_back):
    """
    Integrate ONE child backward in time for N_back steps.
    We do NOT touch the main bundle here.
    Returns the backpropagated (r, p).
    """
    r = child_traj.get_rvec()
    P = child_traj.get_pvec()
    m = child_traj.get_masses()
    s = child_traj.get_state()

    for _ in range(N_back):
        r, P = step_minimal(r, P, m, s,
                            model.A, model.Omega, model.gradU,
                            -dt, hbar=1.0)
    return r, P

def backprop_pair_until_weak_H(bundle,
                               parent_idx,
                               child_idx,
                               dt,
                               model,
                               eps_coup=1.0e-3,
                               kmax=200,
                               check_every=1,
                               verbose=False,
                               name="spawn-H-pair"):
    """
    Backprop *both* the parent and the child TBFs together by -dt
    until the actual TDSE coupling

        | (H - i Sdot)_{parent, child} |

    drops below eps_coup.

    This is the "rewind the 2-TBF micro-world" version, not the
    "drag only the child" version.

    We mutate the bundle in-place (we have to, to rebuild S/H/Sdot
    consistently).
    """

    # ----- grab traj objects -----
    parent_traj = bundle.trajectorylist[parent_idx]
    child_traj  = bundle.trajectorylist[child_idx]

    # masses & states may differ
    m_parent = parent_traj.get_masses()
    m_child  = child_traj.get_masses()
    s_parent = parent_traj.get_state()
    s_child  = child_traj.get_state()

    # 0) measure coupling right now
    K_pc, mag0 = measure_tdse_coupling(bundle, parent_idx, child_idx)

    if verbose:
        print(f"[{name}] start")
        print(f"  parent idx={parent_idx}, state={s_parent}, R={parent_traj.get_rvec()}, P={parent_traj.get_pvec()}")
        print(f"  child  idx={child_idx}, state={s_child},  R={child_traj.get_rvec()},  P={child_traj.get_pvec()}")
        print(f"  initial |K_pc| = {mag0:.6e} (target < {eps_coup:.2e})")

    first_mag = mag0

    for k in range(kmax):
        # 1) backprop parent one step
        r_p = parent_traj.get_rvec()
        P_p = parent_traj.get_pvec()
        r_p_new, P_p_new = step_minimal(
            r_p,
            P_p,
            m_parent,
            s_parent,
            model.A,
            model.Omega,
            model.gradU,
            -dt,   # backward
            hbar=1.0
        )
        parent_traj.set_rvec(r_p_new)
        parent_traj.set_pvec(P_p_new)

        # 2) backprop child one step
        r_c = child_traj.get_rvec()
        P_c = child_traj.get_pvec()
        r_c_new, P_c_new = step_minimal(
            r_c,
            P_c,
            m_child,
            s_child,
            model.A,
            model.Omega,
            model.gradU,
            -dt,   # backward
            hbar=1.0
        )
        child_traj.set_rvec(r_c_new)
        child_traj.set_pvec(P_c_new)

        # 3) only check every N steps
        if (k % check_every) == 0:
            K_pc, mag = measure_tdse_coupling(bundle, parent_idx, child_idx)

            if verbose:
                print(f"  step {k:4d}: |K_pc| = {mag:.6e}")
                print(f"           parent R={parent_traj.get_rvec()}, P={parent_traj.get_pvec()}")
                print(f"           child  R={child_traj.get_rvec()},  P={child_traj.get_pvec()}")

            if mag < eps_coup:
                if verbose:
                    ratio = mag / max(first_mag, 1e-30)
                    print(f"[{name}] stop at k={k}: |K|={mag:.6e} < {eps_coup:.6e}  (ratio={ratio:.3e})")
                return (parent_traj.get_rvec(), parent_traj.get_pvec(),
                        child_traj.get_rvec(),  child_traj.get_pvec(),
                        k, mag)

    # hit kmax
    K_pc, mag = measure_tdse_coupling(bundle, parent_idx, child_idx)
    if verbose:
        ratio = mag / max(first_mag, 1e-30)
        print(f"[{name}] hit kmax={kmax}, final |K|={mag:.6e}  (ratio={ratio:.3e})")
    return (parent_traj.get_rvec(), parent_traj.get_pvec(),
            child_traj.get_rvec(),  child_traj.get_pvec(),
            kmax, mag)

def backprop_bundle_until_weak_H(bundle,
                                 parent_idx,
                                 child_idx,
                                 dt,
                                 model,
                                 eps_coup=1.0e-6,
                                 kmax=200,
                                 check_every=1,
                                 verbose=False,
                                 name="spawn-H-bundle"):
    """
    Make a FULL snapshot of the bundle, rewind EVERY trajectory together
    until |(H - i Sdot)_{p,c}| < eps_coup, then return:
        snapshot_bundle, k_back, last_mag
    You then replay +dt k_back times on the *real* bundle.
    """
    # 1) snapshot current bundle
    snap = copy.deepcopy(bundle)

    # we work on the snapshot
    for k in range(kmax):
        # build couplings at this snapshot time
        snap.BuildHeff()
        K = snap.H - 1j * snap.SDot
        mag = abs(K[parent_idx, child_idx])

        if verbose and (k % check_every == 0):
            print(f"[{name}] k={k:4d}, |K_pc|={mag:.6e}")

        if mag < eps_coup:
            if verbose:
                print(f"[{name}] stop at k={k}, |K|={mag:.6e} < {eps_coup:.6e}")
            return snap, k, mag

        # otherwise: step EVERY traj backward
        for t in snap.trajectorylist:
            r = t.get_rvec()
            P = t.get_pvec()
            m = t.get_masses()
            s = t.get_state()
            r_new, P_new = step_minimal(
                r, P, m, s,
                model.A, model.Omega, model.gradU,
                -dt,  # backward
                hbar=1.0
            )
            t.set_rvec(r_new)
            t.set_pvec(P_new)

    # hit kmax
    snap.BuildHeff()
    K = snap.H - 1j * snap.SDot
    mag = abs(K[parent_idx, child_idx])
    if verbose:
        print(f"[{name}] hit kmax={kmax}, final |K|={mag:.6e}")
    return snap, kmax, mag


def insert_child(bundle, r_child, P_child, state_child, amp=0.0+0.0j):
    """
    Add the new trajectory to the bundle and extend the coefficient vector.
    Assumes bundle.add_trajectory(...) creates the TBF *after* existing ones.
    """
    bundle.add_trajectory(r_child, P_child, state_child)
    C = bundle.GetC()
    # append amplitude at the end
    C[-1] = amp
    bundle.SetC(C)
    # rebuild S etc. after structure change
    bundle.BuildHeff()
    return bundle.ntraj - 1   # index of new child

def measure_tdse_coupling(bundle, parent_idx, child_idx):
    """
    Build S, H, Sdot on the *current* bundle state and return
        K_ij = (H_ij - 1j * Sdot_ij)
    and its magnitude.

    Assumes bundle.KE, bundle.PE, bundle.S, bundle.SDot exist after Build*.
    """
    bundle.BuildHeff()

    K = bundle.H - 1j * bundle.SDot     # this is the TDSE/working coupling

    val = K[parent_idx, child_idx]
    return val, abs(val)

