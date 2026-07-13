import numpy as np
from bundle import Bundle
from models import BerryModel2DParallelTransport
from classprop import step_minimal   # your version
from propagate import PropH          # your exp(-iHdt) in orthonormal frame
from spawn_ot import *
import cma

# Opt helper
def run_cma_spawn(obj, verbose=True):
    x0 = np.array([0.0, 0.0, 0.0])
    best_x = x0.copy()
    best_f = 1e30

    sigma0 = 0.1
    pop = 30
    n_restart = 3

    for r in range(n_restart):
        opts = {
            'verb_disp': 1 if verbose else 0,
            'maxiter': 200,
            'maxfevals': 8000,
            'tolfun': 1e-9,
            'tolfunhist': 1e-9,
            'tolx': 1e-7,
            'popsize': pop,
            'bounds': [[-0.2, -0.2, -np.pi],
                       [ 0.2,  0.2,  np.pi]],
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        while not es.stop():
            X = es.ask()
            F = []
            for x in X:
                fval = obj(np.asarray(x))
                F.append(fval)
                if fval < best_f:
                    best_f = fval
                    best_x = np.asarray(x)
            es.tell(X, F)
            if verbose and es.countiter % 5 == 0:
                print(f"[CMA r={r}] it={es.countiter}, fbest={best_f:.6e}, xbest={best_x}")

        if verbose:
            print(f"[CMA r={r}] done, best_f={best_f:.6e}, best_x={best_x}")

        # prep next restart
        pop = int(pop * 1.6)
        sigma0 *= 0.6

    return best_x, best_f

# --------------------------------------------------
# 1) one-step TDSE + nuclei
# --------------------------------------------------
def advance_one_step(bundle, dt, nslice=20, move_nuclei=True):
    """
    - build Heff = H - i Sdot
    - propagate coefficients with exp(-i Heff dt)
    - optionally move nuclei classically
    returns (C_new, phys_norm)
    """
    model = bundle.model

    # build everything once
    bundle.BuildHeff()

    C_t = np.asarray(bundle.GetC(), dtype=np.complex128)

    # coeffs in non-orthonormal basis, but PropH already handles Heff
    C_new = PropH(bundle.Heff, C_t, dt, nslice, renormalize=False)
    bundle.SetC(C_new)

    if move_nuclei:
        for traj in bundle.trajectorylist:
            r_k = traj.get_rvec()
            P_k = traj.get_pvec()
            st  = traj.get_state()
            m_k = traj.get_masses()

            r_next, P_next = step_minimal(
                r_k, P_k, m_k, st,
                model.A, model.Omega, model.gradU,
                dt, hbar=1.0
            )
            traj.set_rvec(r_next)
            traj.set_pvec(P_next)

    # physical norm
    bundle.BuildS()
    C_new = bundle.GetC()
    phys_norm = np.vdot(C_new, bundle.S @ C_new).real
    return C_new, phys_norm

# --------------------------------------------------
# 2) CMA-ES driven spawn for a single parent
# --------------------------------------------------
def spawn_from_parent(bundle,
                      parent_idx,
                      target_state,
                      dt,
                      rng,
                      cma_sigma=0.1,
                      cma_maxiter=500,
                      coup_tol_back=1e-5,
                      verbose=True):
    """
    - run CMA-ES on (dx, dy, theta) to maximize |H_pc|
    - insert child
    - backprop child until |(H - iSdot)_{pc}| < coup_tol_back
    - forward-prop it back to present
    returns child_idx
    """
    model = bundle.model
    parent_traj = bundle.trajectorylist[parent_idx]
    mass = parent_traj.get_masses()

    # 1) make objective tied to THIS parent
    obj = make_spawn_objective(parent_traj, model, target_state)

    best_x, best_f = run_cma_spawn(obj)

    # decode best_x → R_child, P_child
    dx, dy, theta = best_x
    R_p = parent_traj.get_rvec()
    R_child = R_p + np.array([dx, dy], float)
    u_dir   = np.array([np.cos(theta), np.sin(theta)], float)

    # energy-match momentum
    E_parent = model.classical_energy(R_p,
                                      parent_traj.get_pvec(),
                                      mass,
                                      parent_traj.get_state())

    Phi_parent = (
        model.V_adiabatic(R_p, parent_traj.get_state())
        + model.D_off(R_p, mass, parent_traj.get_state())
        + model.div_d(R_p, mass, parent_traj.get_state(), parent_traj.get_state())
    ).real

    P_child = energy_matched_canonical_momentum(
        R_child,
        u_dir,
        model,
        mass,
        E_parent,
        Phi_parent,
        target_state
    )
    if P_child is None:
        if verbose:
            print("[spawn] energy match failed, skipping child")
        return None

    # 2) insert child with zero amplitude
    child_idx = insert_child(bundle,
                             R_child,
                             P_child,
                             target_state,
                             amp=0.0+0.0j)
    if verbose:
        print(f"[spawn] inserted child at idx={child_idx} on state {target_state}, R={R_child}, P={P_child}")

    # 3) backprop until weak REAL coupling
    snap, k_back, mag_final = backprop_bundle_until_weak_H(
        bundle,
        parent_idx=parent_idx,
        child_idx=child_idx,
        dt=dt,
        model=model,
        eps_coup=coup_tol_back,
        kmax=5000,
        check_every=1,
        verbose=verbose,
        name=f"spawn({parent_idx}->{child_idx})-bundle"
    )

    # At this point, bundle already has BOTH parent and child
    # rewound kbp steps in time.

    # ---------------------------------------------------------------------
    # 5) forward-replay BOTH parent and child to bring them back to "now"
    # ---------------------------------------------------------------------
    parent_traj = bundle.trajectorylist[parent_idx]
    child_traj  = bundle.trajectorylist[child_idx]
    mass_parent  = parent_traj.get_masses()
    mass_child   = child_traj.get_masses()

    for stepback in range(k_back):
        # this moves ALL trajectories + propagates coefficients
        print(f"[spawn] Propagating trajectory forward, we are {k_back - stepback} steps away from where we were")
        advance_one_step(bundle, dt, nslice=20, move_nuclei=True)

    # 6) rebuild S etc. so the TDSE step right after this sees the updated config
    bundle.BuildS()
    if verbose:
        print(f"[spawn] done, k_back = {k_back}, final child R={child_traj.get_rvec()}, P={child_traj.get_pvec()}")

    return child_idx

# --------------------------------------------------
# 3) MAIN SIMULATION LOOP
# --------------------------------------------------
def run_sim_with_spawning():
    ndim = 2
    model = BerryModel2DParallelTransport(0.02, 3.0, 5.0)
    bundle = Bundle(ndim, model)

    # --- initial TBF: excited, at (-3,0), canonical P=(20,0)
    init_rvec = np.asarray([-3.0, 0.0])
    init_pvec = np.asarray([20.0, 0.0])   # canonical
    init_state = 1                        # excited
    bundle.add_trajectory(init_rvec, init_pvec, init_state)

    # coefficient: just 1.0
    bundle.BuildS()
    C = bundle.GetC()
    # ensure C^† S C = 1
    nS = np.vdot(C, bundle.S @ C).real
    C /= np.sqrt(nS)
    bundle.SetC(C)

    # --- time loop
    h = 1.0e-2
    nsteps = 20000
    renorm_every = 10
    spawn_check_every = 5    # don't try to spawn every single step
    coupling_spawn_thresh = 2.0e-2
    rng = np.random.default_rng(1236)
    curr = 0

    print(f"h = {h}")
    print("Before propagation")
    bundle.BuildS()
    C = bundle.GetC()
    print("C^† S C =", np.vdot(C, bundle.S @ C))

    for step in range(nsteps):
        # 1) TDSE + nuclei
        C_new, phys_norm = advance_one_step(bundle, h, nslice=20, move_nuclei=True)

        # 2) maybe renormalize coefficients
        if (step % renorm_every) == 0:
            bundle.BuildS()
            C = bundle.GetC()
            nS = np.vdot(C, bundle.S @ C).real
            C /= np.sqrt(nS)
            bundle.SetC(C)
            if step % 100 == 0:
                print(f"[step {step}] renorm: C^† S C -> 1.0")

        # 3) spawning check
        if (step % spawn_check_every) == 0:
            # which parents want to spawn?
            cand = should_spawn(bundle,
                                model,
                                step,
                                thresh=coupling_spawn_thresh,
                                cooldown_steps=curr,
                                verbose=True)
            if len(cand) > 0:
                print(f"[step {step}] spawn triggers: {cand}")
                curr = np.inf # Spawn once all simulation

            # if 2+ triggers at same step, do them in order of parent index
            for (parent_idx, target_state) in cand:
                # avoid infinite explosion: limit total trajs
                if bundle.ntraj > 30:
                    print("[spawn] max trajs reached, skipping more spawns")
                    break

                child_idx = spawn_from_parent(
                    bundle,
                    parent_idx,
                    target_state,
                    dt=h,
                    rng=rng,
                    cma_sigma=0.05,
                    cma_maxiter=20,
                    coup_tol_back=1.0e-5,
                    verbose=True
                )
                if child_idx is not None:
                    # mark parent
                    bundle.spawn_meta[parent_idx]["last_spawn_step"] = step
                    bundle.spawn_meta[parent_idx]["last_spawn_target"] = target_state
                    # initialize child’s meta so it doesn’t immediately spawn back
                    parent_traj = bundle.trajectorylist[parent_idx]
                    bundle.spawn_meta[child_idx] = {
                        "last_spawn_step": step,
                        "last_spawn_target": parent_traj.get_state()
                    }


        # 4) (optional) diagnostics
        if (step % 50) == 0:
            # print first traj's canonical and mechanical momenta
            for idx in range(bundle.ntraj):
                t0 = bundle.trajectorylist[idx]
                r0 = t0.get_rvec()
                P0 = t0.get_pvec()
                st0 = t0.get_state()
                A0 = model.A(r0, st0)
                p_mech0 = P0 + A0
                Eclass0 = model.classical_energy(r0, P0, t0.get_masses(), st0)
                print(f"[step {step}, traj {idx}] r0={r0}, P0={P0}, p_mech0={p_mech0}, Eclass0={Eclass0}")
            print(f"[step {step}] #traj = {bundle.ntraj}")
            C = bundle.GetC()
            print(f"[step {step}] XS pop =", abs(C[0])**2)
            if len(bundle.trajectorylist) == 2:
                print(f"[step {step}] GS pop =", abs(C[1])**2)

    print("After propagation")
    bundle.BuildS()
    C = bundle.GetC()
    print("final C^† S C =", np.vdot(C, bundle.S @ C).real)
    print("final C=", C)
    print("final XS pop =", abs(C[0])**2)
    print("final GS pop =", abs(C[1])**2)
    print("#final traj =", bundle.ntraj)


if __name__ == "__main__":
    run_sim_with_spawning()

