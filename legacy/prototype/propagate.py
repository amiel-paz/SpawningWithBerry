import copy
import sys
import math
import os
import re
import numpy as np
import cmath
import random
from matutils import *
from classprop import step_minimal

def PropH(H,C_t,dt,nslice,start=0,renormalize=False):
    I = np.eye(C_t.shape[0],C_t.shape[0])
    B = -1j * H * dt
    C_tdt_prev = np.zeros(C_t.shape,dtype=complex)
    for n in range(start,nslice):
        Bn = B / 2**n
        Bn2 = Bn @ Bn
        Bn3 = Bn @ Bn2
        Bn4 = Bn2 @ Bn2
        tayl = I + Bn + Bn2/2.0 + Bn3/6.0 + Bn4/24.0
        for j in range(n):
            tayl = tayl @ tayl
        C_tdt = tayl @ C_t
        error = np.sqrt(np.sum(np.abs(C_tdt - C_tdt_prev)**2))
        if (error < 1.0e-10):
            break
        C_tdt_prev = C_tdt
    if renormalize:
        C_tdt *= 1/np.linalg.norm(C_tdt)
    return C_tdt

def energy_orthonormal(bundle, C):
    # assumes bundle.BuildS() and BuildSp5inv() already called this step
    y = bundle.Sp5inv @ C                             # y = S^{1/2} C
    Htilde = bundle.Sp5inv.conj().T @ bundle.H @ bundle.Sp5inv
    return float(np.vdot(y, Htilde @ y).real)

def s_norm(bundle, C):
    return float(np.sqrt(np.vdot(C, bundle.S @ C).real))

def s_normalize(bundle, C):
    n = s_norm(bundle, C)
    return C if n == 0 else C / n

def energy_monitor_step(bundle, C_prev, C_curr, dt, nslice,
                        tau_rev=1e-11, denom_floor=1.0):
    # Make sure S, Sp5inv, H are current
    bundle.BuildS(); bundle.BuildSp5inv(); bundle.BuildH()

    # normalize in S-metric (as you already do)
    C_prev_n = s_normalize(bundle, C_prev)
    C_curr_n = s_normalize(bundle, C_curr)

    E_prev = energy_orthonormal(bundle, C_prev_n)
    E_curr = energy_orthonormal(bundle, C_curr_n)

    # reversibility check on the *current* generator
    rev = reversibility_defect(bundle, C_prev_n, dt, nslice)

    dE_abs = abs(E_curr - E_prev)
    dE_rel = dE_abs / max(denom_floor, abs(E_prev))

    return {"E_prev":E_prev, "E_curr":E_curr,
            "dE_abs":dE_abs, "dE_rel":dE_rel, "rev_defect":rev}


def reversibility_defect(bundle, C, dt, nslice):
    # forward
    Cf = PropH(bundle.Heff, C, +dt, nslice, renormalize=False)
    # backward (reuse same Heff; Sdot flips sign automatically next build, but
    # for a pure propagator test we just apply -dt to the same generator)
    Cb = PropH(bundle.Heff, Cf, -dt, nslice, renormalize=False)
    # S-metric distance
    # (don’t renormalize the probe; we want the pure integrator effect)
    v = Cb - C
    return float(np.sqrt(np.vdot(v, bundle.S @ v).real))

def step_cayley(bundle, dt, eig_thresh=1e-8, regularize=0.0):
    """
    Metric-orthonormal Cayley step:
      y = S^{1/2} C
      Htil = S^{-1/2} (H - i * SDot / 2) S^{-1/2}   (Hermitian)
      (I + i dt/2 Htil) y' = (I - i dt/2 Htil) y
      C' = S^{-1/2} y'
    Everything is built locally; bundle's builders remain unchanged.
    """
    # --- Build raw operators (as-is) ---
    bundle.BuildS()
    bundle.BuildH()
    bundle.BuildSDot()
    S  = 0.5*(bundle.S + bundle.S.conj().T)         # enforce Hermitian numerically
    Hh = 0.5*(bundle.H + bundle.H.conj().T)         # Hermitian part
    dS = 0.5*(bundle.SDot + bundle.SDot.conj().T)   # Hermitian dotS

    # --- Eigen decomp of S and subspace selection ---
    lam, U = np.linalg.eigh(S)                      # S = U diag(lam) U†
    keep = lam > eig_thresh
    if not np.any(keep):
        raise RuntimeError("All S eigenvalues fell below threshold.")
    U_r   = U[:, keep]
    lam_r = lam[ keep]
    # condition log
    # print(f"[orth Cayley] keep {keep.sum()}/{len(lam)} modes; cond ~ {lam_r.max()/lam_r.min():.2e}")

    # S^{±1/2} on the kept subspace
    Lmh = np.diag(lam_r**(-0.5))                    # Λ^{-1/2}
    Lph = np.diag(lam_r**( +0.5))                   # Λ^{+1/2}
    S_mhalf = U_r @ Lmh @ U_r.conj().T              # S^{-1/2} (pseudoinverse on subspace)
    S_phalf = U_r @ Lph @ U_r.conj().T              # S^{+1/2}  (projector-consistent)

    # --- Transform H_eff to orthonormal frame, enforce Hermitian ---
    Heff_phys = Hh - 0.5j * dS
    Htil = S_mhalf.conj().T @ Heff_phys @ S_mhalf
    Htil = 0.5 * (Htil + Htil.conj().T)             # force Hermitian numerically

    # --- Cayley in the y-frame (I-metric) ---
    I = np.eye(Htil.shape[0], dtype=np.complex128)
    if regularize and regularize > 0.0:
        Ireg = regularize * I
    else:
        Ireg = 0.0

    A = I + 0.5j * dt * Htil + Ireg
    B = I - 0.5j * dt * Htil + Ireg

    # current C, map to y
    C  = bundle.GetC()
    y  = S_phalf @ C                                 # y = S^{1/2} C
    yp = np.linalg.solve(A, B @ y)                   # unitary up to solve tol

    # map back: C' = S^{-1/2} y' (consistent subspace)
    Cp = S_mhalf @ yp
    bundle.SetC(Cp)

    # pre-move norm in the SAME metric used for build (should be ~1)
    # (optional, for your sanity prints)
    norm_pre = float(np.vdot(Cp, (U_r @ np.diag(lam_r) @ U_r.conj().T) @ Cp).real)
    return Cp

def advance_one_step(bundle, dt, nslice=20, move_nuclei=True, renormalize=True):
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

    if renormalize:
        C_new /= math.sqrt(phys_norm)
        bundle.SetC(C_new)
        phys_norm = np.vdot(C_new, bundle.S @ C_new).real

    bundle.BuildHeff()
    return C_new, phys_norm

# The recursive Crank-Nicholson solver
import numpy as np
from typing import Tuple, Dict, Any, List
from contextlib import contextmanager

Array = np.ndarray

# ---------------------- safe BuildHeff snapshot ----------------------

@contextmanager
def _temporary_phase_space(bundle, R_list: List[Array], P_list: List[Array]):
    R_save, P_save = [], []
    try:
        for i, traj in enumerate(bundle.trajectorylist):
            R_save.append(np.array(traj.rvec, copy=True))
            P_save.append(np.array(traj.pvec, copy=True))
            traj.set_rvec(np.array(R_list[i], copy=False))
            traj.set_pvec(np.array(P_list[i], copy=False))
        yield
    finally:
        for i, traj in enumerate(bundle.trajectorylist):
            traj.set_rvec(R_save[i])
            traj.set_pvec(P_save[i])

def _build_Heff_at(bundle, R_list: List[Array], P_list: List[Array]):
    with _temporary_phase_space(bundle, R_list, P_list):
        bundle.BuildHeff()
        S = np.array(bundle.S, copy=True)
        H = np.array(bundle.H, copy=True)
        tau = np.array(bundle.Sdot, copy=True)   # asymmetric part
    return S, H, tau

# ---------------------- frozen linear nuclear path (relative time) ----------------------

def _linear_path_relative(
    R_a: List[Array], P_a: List[Array],
    R_b: List[Array], P_b: List[Array],
    Tseg: float
):
    """
    Parameterize the frozen nuclear path on θ ∈ [0,1]:
        R(θ) = (1-θ) R_a + θ R_b
        P(θ) = (1-θ) P_a + θ P_b
    Absolute time never appears. Actual substep length is dt = θ_span * Tseg.
    """
    Ra = [np.array(r, copy=True) for r in R_a]
    Pa = [np.array(p, copy=True) for p in P_a]
    Rb = [np.array(r, copy=True) for r in R_b]
    Pb = [np.array(p, copy=True) for p in P_b]

    def R_of_theta(theta: float) -> List[Array]:
        return [(1.0 - theta) * Ra[i] + theta * Rb[i] for i in range(len(Ra))]

    def P_of_theta(theta: float) -> List[Array]:
        return [(1.0 - theta) * Pa[i] + theta * Pb[i] for i in range(len(Pa))]

    return R_of_theta, P_of_theta, float(Tseg)

# ---------------------- S-skew projection helpers ----------------------

def _project_K_to_S_skew(Sm: Array, Km: Array, jitter: float = 0.0) -> Tuple[Array, Array]:
    Delta = Km.conj().T @ Sm + Sm @ Km
    Sm_eff = Sm + jitter * np.eye(Sm.shape[0], dtype=Sm.dtype) if jitter > 0.0 else Sm
    X = np.linalg.solve(Sm_eff, Delta)
    Ktilde = Km - 0.5 * X
    return Ktilde, X

def _norm_S_invariant(S: Array, X: Array, which: str = "fro") -> float:
    try:
        L = np.linalg.cholesky(S)
        Linv = np.linalg.inv(L)
        Y = L @ X @ Linv
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(S)
        eps = max(1e-14, 1e-12 * float(np.max(w)))
        keep = w > eps
        V1 = V[:, keep]; w1 = w[keep]
        Shalf = V1 @ (np.sqrt(w1)[:, None] * V1.T.conj())
        Shalf_inv = V1 @ ((1.0 / np.sqrt(w1))[:, None] * V1.T.conj())
        Y = Shalf @ X @ Shalf_inv
    if which == "fro":
        return float(np.linalg.norm(Y, "fro"))
    if which == "2":
        return float(np.linalg.svd(Y, compute_uv=False, hermitian=False)[0])
    raise ValueError("which must be 'fro' or '2'")

# ---------------------- one CN substep at θ-midpoint ----------------------

def _cn_substep(
    Ca: Array,
    Sa: Array, Sb: Array,
    Sm: Array, Hm: Array, taum: Array,
    dt: float,
    *,
    use_projection: bool,
    proj_guard_eta: float,
    S_projection_jitter: float = 0.0,
) -> Tuple[Array, float, float]:
    Km = Hm - 1j * taum
    proj_size = 0.0
    if use_projection:
        Ktilde, X = _project_K_to_S_skew(Sm, Km, jitter=S_projection_jitter)
        proj_size = _norm_S_invariant(Sm, X, which="fro")
        if proj_size <= proj_guard_eta:
            Km = Ktilde

    A = Sm - 0.5 * dt * Km
    B = Sm + 0.5 * dt * Km
    Cb = np.linalg.solve(A, B @ Ca)

    nu_a = np.real_if_close(Ca.conj().T @ (Sa @ Ca))
    nu_b = np.real_if_close(Cb.conj().T @ (Sb @ Cb))
    denom = float(nu_a) if abs(nu_a) > 0 else 1.0
    drift_ratio = abs(float(nu_b) / denom - 1.0)
    return Cb, drift_ratio, proj_size

# ---------------------- recursive adaptive segment stepper (relative) ----------------------

def evolve_C_on_segment_relative(
    bundle,
    C_init: Array,
    R_a: List[Array], P_a: List[Array],
    R_b: List[Array], P_b: List[Array],
    Tseg: float,
    *,
    eC_tol: float = 1e-3,
    dt_min: float = 1e-8,
    max_depth: int = 20,
    use_projection: bool = True,
    proj_guard_eta: float = 1e-6,
    S_projection_jitter: float = 0.0,
) -> Tuple[Array, Dict[str, Any]]:
    """
    Adaptive CN over a *relative* interval of length Tseg, with θ ∈ [0,1].
    Subinterval [θa, θb] has physical duration dt = (θb-θa)*Tseg.
    """
    R_of_theta, P_of_theta, Tseg = _linear_path_relative(R_a, P_a, R_b, P_b, Tseg)

    S_cache: Dict[float, Array] = {}
    H_cache: Dict[float, Array] = {}
    T_cache: Dict[float, Array] = {}

    def S_at(theta: float) -> Array:
        if theta not in S_cache:
            S_cache[theta], _, _ = _build_Heff_at(bundle, R_of_theta(theta), P_of_theta(theta))
        return S_cache[theta]

    def Htau_at_mid(theta: float) -> Tuple[Array, Array, Array]:
        if theta not in S_cache or theta not in H_cache or theta not in T_cache:
            S_m, H_m, tau_m = _build_Heff_at(bundle, R_of_theta(theta), P_of_theta(theta))
            S_cache[theta] = S_m; H_cache[theta] = H_m; T_cache[theta] = tau_m
        return S_cache[theta], H_cache[theta], T_cache[theta]

    solves = rejects = leaves = 0
    max_local_drift = max_proj_size = 0.0
    reached_depth = 0

    def advance(theta_a: float, Ca: Array, theta_b: float, depth: int) -> Array:
        nonlocal solves, rejects, leaves, max_local_drift, max_proj_size, reached_depth
        reached_depth = max(reached_depth, depth)
        dtheta = theta_b - theta_a
        theta_m = 0.5 * (theta_a + theta_b)
        dt = dtheta * Tseg

        Sa = S_at(theta_a)
        Sb = S_at(theta_b)
        Sm, Hm, taum = Htau_at_mid(theta_m)

        Cb, drift, proj_size = _cn_substep(
            Ca, Sa, Sb, Sm, Hm, taum, dt,
            use_projection=use_projection,
            proj_guard_eta=proj_guard_eta,
            S_projection_jitter=S_projection_jitter,
        )
        solves += 1
        max_local_drift = max(max_local_drift, drift)
        max_proj_size   = max(max_proj_size, proj_size)

        if drift <= eC_tol or (dt <= dt_min) or (depth >= max_depth):
            leaves += 1
            return Cb

        rejects += 1
        Cmid = advance(theta_a, Ca, theta_m, depth + 1)
        Cend = advance(theta_m, Cmid, theta_b, depth + 1)
        return Cend

    C_final = advance(0.0, C_init, 1.0, depth=0)
    stats = dict(
        solves=solves,
        rejected=rejects,
        leaves=leaves,
        max_local_drift=max_local_drift,
        max_proj_size=max_proj_size,
        max_depth=reached_depth,
    )
    return C_final, stats

# ---------------------- convenience wrappers: left and right halves ----------------------

def evolve_C_left_half(
    bundle,
    C_init: Array,
    R_init: List[Array], P_init: List[Array],
    R_mid:  List[Array], P_mid:  List[Array],
    T_half: float,
    **kwargs
) -> Tuple[Array, Dict[str, Any]]:
    """Evolve C from (init) to (mid) over a relative duration T_half (no absolute time)."""
    return evolve_C_on_segment_relative(bundle, C_init, R_init, P_init, R_mid, P_mid, T_half, **kwargs)

def evolve_C_right_half(
    bundle,
    C_mid: Array,
    R_mid:  List[Array], P_mid:  List[Array],
    R_fin:  List[Array], P_fin:  List[Array],
    T_half: float,
    **kwargs
) -> Tuple[Array, Dict[str, Any]]:
    """Evolve C from (mid) to (final) over a relative duration T_half (no absolute time)."""
    return evolve_C_on_segment_relative(bundle, C_mid, R_mid, P_mid, R_fin, P_fin, T_half, **kwargs)

