import numpy as np
from typing import Callable, Tuple, Dict, Any
Array = np.ndarray

def boris_rotate_v(v, Omega_mat, m, dt, hbar=1.0):
    """
    dv/dt = (hbar/m) * Omega_mat @ v
    Cayley / Boris rotation.
    """
    v = np.asarray(v, dtype=np.complex128)
    Omega_mat = np.asarray(Omega_mat, dtype=np.complex128)

    T = 0.5 * (hbar / m) * dt * Omega_mat
    I = np.eye(v.shape[0], dtype=np.complex128)

    v_rot = np.linalg.solve(I - T, (I + T) @ v)
    return v_rot.real

def step_minimal(r, P, m, state,
                 A_func,        # A_func(r, state) -> A(r)
                 Omega_func,    # Omega_func(r, state) -> Ω(r) = (∇A)^T - (∇A)
                 gradU_func,    # gradU_func(r, m, state) -> ∇Φ(r)
                 dt,
                 hbar=1.0):
    """
    Symmetric Boris-like step for the Hamiltonian

        H = 1/(2m) * (P + hbar A(r))**2 + Φ(r).

    Equations of motion implied by this H are

        v = (P + hbar A)/m
        m dv/dt = - hbar Ω(r) v - ∇Φ(r)
        dr/dt   = v

    Note the **minus** in front of hbar Ω v.
    """

    # promote
    r = np.asarray(r, dtype=np.complex128)
    P = np.asarray(P, dtype=np.complex128)

    # 1) velocity from canonical momentum (plus-A Hamiltonian)
    A_r = np.asarray(A_func(r, state), dtype=np.complex128)
    v   = (P + hbar * A_r) / m

    # 2) scalar force at start
    gradU_r = np.asarray(gradU_func(r, m, state), dtype=np.complex128)
    F0 = -gradU_r     # physical force

    # 3) first half-kick
    v_minus = v + 0.5 * dt * (F0 / m)

    # 4) Berry / magnetic rotation — **with minus sign**
    Omega_r = np.asarray(Omega_func(r, state), dtype=np.complex128)
    # we keep boris_rotate_v as-is (it solves dv/dt = (hbar/m) Ω v),
    # so we just feed it -Ω to get dv/dt = -(hbar/m) Ω v.
    v_rot = boris_rotate_v(v_minus, -Omega_r, m, dt, hbar=hbar)

    # 5) predict position
    r_pred = (r + dt * v_rot).real

    # 6) scalar force at predicted position
    gradU_r1 = np.asarray(gradU_func(r_pred, m, state), dtype=np.complex128)
    F1 = -gradU_r1

    # 7) second half-kick
    v_new = v_rot + 0.5 * dt * (F1 / m)

    # 8) final drift
    r_new = (r + dt * v_new).real

    # 9) rebuild canonical momentum: P = m v - hbar A
    A_r_new = np.asarray(A_func(r_new, state), dtype=np.complex128)
    P_new   = (m * v_new - hbar * A_r_new).real

    return r_new, P_new

def implicit_midpoint_step(
    bundle,
    h: float,
    mid_tol: float = 1e-12,
    max_iters: int = 8,
    damping: float = 1.0,
    predictor: str = "euler",
):
    """
    Implicit-midpoint for *all trajectories* in bundle (R,P only).
    Updates bundle.trajectorylist[i].rvec/pvec to endpoints and returns midpoints too.
    Assumes model methods:
        get_rdot(G, R, P, state [, t]) -> dR/dt
        get_pdot(G, R, P, state [, t]) -> dP/dt
    where G = diag(1/m_i) (or your mass inverse).
    """

    # Snapshot inputs
    traj_Rn, traj_Pn, traj_state, traj_masses = [], [], [], []
    for traj in bundle.trajectorylist:
        traj_Rn.append(np.array(traj.rvec, copy=True))
        traj_Pn.append(np.array(traj.pvec, copy=True))
        traj_state.append(traj.state)
        traj_masses.append(np.array(traj.masses, copy=False))

    # Build initial endpoint guesses
    traj_Rk, traj_Pk = [], []
    for i, (Rn, Pn, state, masses) in enumerate(zip(traj_Rn, traj_Pn, traj_state, traj_masses)):
        G = np.diag(1.0 / masses)
        if predictor == "euler":
            Rprime = bundle.model.get_rdot(G, Rn, Pn, state)
            Pprime = bundle.model.get_pdot(G, Rn, Pn, state)
            traj_Rk.append(Rn + h * Rprime)
            traj_Pk.append(Pn + h * Pprime)
        elif predictor == "keep":
            traj_Rk.append(Rn.copy())
            traj_Pk.append(Pn.copy())
        else:
            raise ValueError("predictor must be 'euler' or 'keep'")

    converged = False
    residual = np.inf

    for it in range(1, max_iters + 1):
        traj_Rk_new, traj_Pk_new = [], []

        for i, (Rn, Pn, state, masses) in enumerate(zip(traj_Rn, traj_Pn, traj_state, traj_masses)):
            Rk = traj_Rk[i]
            Pk = traj_Pk[i]
            G  = np.diag(1.0 / masses)

            # Midpoint in phase space
            Rm = 0.5 * (Rn + Rk)
            Pm = 0.5 * (Pn + Pk)

            # Evaluate RHS at the midpoint (functions return vectors, not callables)
            Rdot_m = bundle.model.get_rdot(G, Rm, Pm, state)
            Pdot_m = bundle.model.get_pdot(G, Rm, Pm, state)

            Rk_new = Rn + h * Rdot_m
            Pk_new = Pn + h * Pdot_m

            if damping != 1.0:
                Rk_new = (1.0 - damping) * Rk + damping * Rk_new
                Pk_new = (1.0 - damping) * Pk + damping * Pk_new

            traj_Rk_new.append(Rk_new)
            traj_Pk_new.append(Pk_new)

        # Convergence check across all trajectories
        Rflat_new = np.concatenate([v.ravel() for v in traj_Rk_new])
        Rflat_old = np.concatenate([v.ravel() for v in traj_Rk])
        Pflat_new = np.concatenate([v.ravel() for v in traj_Pk_new])
        Pflat_old = np.concatenate([v.ravel() for v in traj_Pk])
        residual = max(
            float(np.linalg.norm(Rflat_new - Rflat_old, ord=np.inf)),
            float(np.linalg.norm(Pflat_new - Pflat_old, ord=np.inf)),
        )

        traj_Rk, traj_Pk = traj_Rk_new, traj_Pk_new

        if residual < mid_tol:
            converged = True
            break

    # Finalize outputs (endpoints + midpoints per trajectory)
    traj_Rfinal, traj_Pfinal, traj_Rm, traj_Pm = [], [], [], []
    for i, (Rn, Pn) in enumerate(zip(traj_Rn, traj_Pn)):
        R1 = traj_Rk[i]
        P1 = traj_Pk[i]
        Rm = 0.5 * (Rn + R1)
        Pm = 0.5 * (Pn + P1)

        traj_Rfinal.append(R1)
        traj_Pfinal.append(P1)
        traj_Rm.append(Rm)
        traj_Pm.append(Pm)

        bundle.trajectorylist[i].set_rvec(R1)
        bundle.trajectorylist[i].set_pvec(P1)

    info = {"converged": converged, "iters": it if converged else max_iters, "residual": residual}
    return info, traj_Rfinal, traj_Pfinal, traj_Rm, traj_Pm




def _boris_step_from_v0(v0, F_m_over_m, Omega_m, h, mass_scalar, hbar=1.0):
    v_minus = v0 + 0.5 * h * F_m_over_m
    v_rot   = boris_rotate_v(v_minus, -Omega_m, m=mass_scalar, dt=h, hbar=hbar)
    v_plus  = v_rot + 0.5 * h * F_m_over_m
    return v_plus


def implicit_midpoint_step_boris(
    bundle,
    h: float,
    mid_tol: float = 1e-12,
    max_iters: int = 12,
    damping: float = 1.0,
    hbar: float = 1.0,
):
    """
    Boris-consistent implicit-midpoint for all trajectories’ (R,P).
    - Iterates R_m (midpoint position); evaluates F_m=-∇Φ and Ω at R_m.
    - Uses Boris (scalar half-kicks + Cayley rotation by -Ω) to obtain v1 from v0.
    - Uses v_m = 0.5*(v0+v1) to enforce midpoint geometry; R_m = R0 + 0.5 h v_m.
    - Final R1 = R0 + h v_m; P1 = m v1 - ℏ A(R1)  (canonical).
    Returns (info, R1_list, P1_list, Rm_list, Pm_list).
    """
    traj_R0, traj_P0, traj_state, traj_m = [], [], [], []
    for traj in bundle.trajectorylist:
        traj_R0.append(np.array(traj.rvec, copy=True, dtype=np.float64))
        traj_P0.append(np.array(traj.pvec, copy=True, dtype=np.float64))
        traj_state.append(traj.state)
        traj_m.append(np.array(traj.masses, copy=False, dtype=np.float64))

    model = bundle.model

    # Initial guess for R_m: Euler from current mechanical v
    Rm = []
    for R0, P0, st, m in zip(traj_R0, traj_P0, traj_state, traj_m):
        A0   = np.asarray(model.A(R0, st), dtype=np.complex128).real
        v0   = (P0 + hbar * A0) / m
        Rm.append(R0 + 0.5 * h * v0)
    Rm = [r.copy() for r in Rm]

    converged = False
    it = 0
    for it in range(1, max_iters + 1):
        Rm_new = []
        max_res = 0.0

        # Per-trajectory update using current midpoint Rm
        R1_list, P1_list, Pm_list = [], [], []
        for i, (R0, P0, st, m, Rm_i) in enumerate(zip(traj_R0, traj_P0, traj_state, traj_m, Rm)):
            # Evaluate midpoint fields
            F_m   = -model.gradU(Rm_i, m, st).real              # scalar force
            Omega = np.asarray(model.Omega(Rm_i, st), dtype=np.complex128).real

            # Build v0 from *start* canonical momentum
            A0  = np.asarray(model.A(R0, st), dtype=np.complex128).real
            v0  = (P0 + hbar * A0) / m

            # Boris update at midpoint fields → v1
            F_over_m = F_m / m
            v1 = _boris_step_from_v0(v0, F_over_m, Omega, h, m, hbar=hbar)

            # Midpoint mechanical velocity and position constraint
            v_m  = 0.5 * (v0 + v1)
            Rm_i_new = R0 + 0.5 * h * v_m

            if damping != 1.0:
                Rm_i_new = (1.0 - damping) * Rm_i + damping * Rm_i_new

            # Endpoint position using midpoint velocity
            R1 = R0 + h * v_m

            # Endpoint canonical momentum from endpoint A(R1) and v1
            A1 = np.asarray(model.A(R1, st), dtype=np.complex128).real
            P1 = (m * v1 - hbar * A1)

            Rm_new.append(Rm_i_new)
            R1_list.append(R1)
            P1_list.append(P1)
            Pm_list.append(0.5 * (P0 + P1))

            # residual on R_m fixed-point
            max_res = max(max_res, float(np.linalg.norm(Rm_i_new - Rm_i, ord=np.inf)))

        Rm = Rm_new
        if max_res < mid_tol:
            converged = True
            # Commit and exit
            for i, (R1, P1) in enumerate(zip(R1_list, P1_list)):
                bundle.trajectorylist[i].set_rvec(R1.astype(float))
                bundle.trajectorylist[i].set_pvec(P1.astype(float))
            # prepare midpoint outputs from the fixed-point iterate
            Rm_out = Rm
            Pm_out = Pm_list
            return (
                {"converged": True, "iters": it, "residual": max_res},
                R1_list, P1_list, Rm_out, Pm_out)

    if not converged:
        # still commit the last iterate
         R1_list, P1_list, Pm_list = [], [], []
         for i, (R0, P0, st, m, Rm_i) in enumerate(zip(traj_R0, traj_P0, traj_state, traj_m, Rm)):
             F_m   = -model.gradU(Rm_i, m, st).real
             Omega = np.asarray(model.Omega(Rm_i, st), dtype=np.complex128).real
             A0    = np.asarray(model.A(R0, st), dtype=np.complex128).real
             v0    = (P0 + hbar * A0) / m
             v1    = _boris_step_from_v0(v0, F_m/m, Omega, h, hbar=hbar)
             v_m   = 0.5 * (v0 + v1)
             R1    = R0 + h * v_m
             A1    = np.asarray(model.A(R1, st), dtype=np.complex128).real
             P1    = (m * v1 - hbar * A1)
             bundle.trajectorylist[i].set_rvec(R1.astype(float))
             bundle.trajectorylist[i].set_pvec(P1.astype(float))
             R1_list.append(R1)
             P1_list.append(P1)
             Pm_list.append(0.5 * (P0 + P1))

    info = {"converged": converged, "iters": it, "residual": max_res if 'max_res' in locals() else None}
    # Rm are the last fixed-point iterates; Pm are the arithmetic midpoints of canonical P
    Rm_out = Rm
    Pm_out = Pm_list
    return info, R1_list, P1_list, Rm_out, Pm_out
