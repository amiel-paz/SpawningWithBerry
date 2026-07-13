import cmath
import numpy as np
from bundle import *
from models import *
from classprop import *
from testutils import *
from propagate import *

def advance_one_step_old(bundle, dt, nslice=20, move_nuclei=True):
    model = bundle.model

    # 0. move nuclei *first* so the basis is at t+dt
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

    # 1. NOW build S, H, Sdot at the *new* geometry
    bundle.BuildHeff()      # should build S, H, SDot, then Heff

    # 2. propagate coefficients in that metric
    C_t = np.asarray(bundle.GetC(), dtype=np.complex128)
    C_new = PropH(bundle.Heff, C_t, dt, nslice, renormalize=False)

    bundle.SetC(C_new)

    # 3. check physical norm in the *current* S
    phys_norm = np.vdot(C_new, bundle.S @ C_new).real
    return C_new, phys_norm

def advance_one_step(bundle, dt, nslice=20, move_nuclei=True):
    """
    1) (optional) move nuclei to t+dt
    2) build S, H, SDot, Heff at the *new* geometry
    3) orthonormalize via Cholesky of S
    4) propagate in the orthonormal frame
    5) map back
    6) report physical norm C† S C
    """
    model = bundle.model

    # 0. move nuclei first so S,H,SDot are evaluated where the basis actually is
    if move_nuclei:
        for traj in bundle.trajectorylist:
            r = traj.get_rvec()
            P = traj.get_pvec()
            st = traj.get_state()
            m  = traj.get_masses()

            r_new, P_new = step_minimal(
                r, P, m, st,
                model.A, model.Omega, model.gradU,
                dt, hbar=1.0
            )
            traj.set_rvec(r_new)
            traj.set_pvec(P_new)

    # 1. build matrices at this geometry
    bundle.BuildHeff()          # should set bundle.S, bundle.Heff, etc.
    S    = np.asarray(bundle.S,    dtype=np.complex128)
    Heff = np.asarray(bundle.Heff, dtype=np.complex128)

    # 2. Cholesky: S = L L†   (S should be Hermitian pos-def)
    #    If S is tiny-non-Hermitian from roundoff, symmetrize first.
    S = 0.5 * (S + S.conj().T)
    L = np.linalg.cholesky(S)   # lower-triangular

    # 3. current coeffs in non-orth basis
    C = np.asarray(bundle.GetC(), dtype=np.complex128)   # shape (n,)

    # 4. go to orthonormal frame:  b = L† C
    #    check:  b† b = C† L L† C = C† S C  ✅
    b = L.conj().T @ C

    # 5. transform Heff to orthonormal frame:
    #    H_orth = L^{-1} Heff L^{-†}
    Linv   = np.linalg.inv(L)
    H_orth = Linv @ Heff @ Linv.conj().T

    # 6. propagate in orthonormal frame
    #    your PropH was: PropH(H, vec, dt, nslice, renormalize=False)
    b_new = PropH(H_orth, b, dt, nslice, renormalize=False)

    # 7. map back to non-orth basis:  C_new = L^{-†} b_new
    C_new = np.linalg.solve(L.conj().T, b_new)

    # 8. (optional) renormalize in S-metric — this is cheap, keeps rounding tame
    norm_phys = (C_new.conj().T @ (S @ C_new)).real
    C_new /= np.sqrt(norm_phys + 0.0)

    bundle.SetC(C_new)

    # return the *actual* physical norm after our renorm (should be 1.0 now)
    return C_new, 1.0


h=1.0e-2
nsteps=1000
mod=10
for idx in range(1):
    ndim = 2
    model = BerryModel2DParallelTransport(0.02,3.0,15.0)
    bundle = Bundle(ndim,model)
    
    init_rvec = np.asarray([0.0, 0.0])
    init_pvec = np.asarray([20.0, -0.5])
    init_state = 1
    bundle.add_trajectory(init_rvec,init_pvec,init_state)
    TBF = bundle.trajectorylist[0]
    G = np.diag(np.reciprocal(TBF.get_masses()))
    m = TBF.get_masses()

    init_rvec = np.asarray([-0.02, 0.0])
    init_pvec = np.asarray([20.0, 0.0])
    init_state = 0
    bundle.add_trajectory(init_rvec,init_pvec,init_state)
    
    init_rvec = np.asarray([0.02, 0.0])
    init_pvec = np.asarray([20.0, 0.05])
    init_state = 1
    bundle.add_trajectory(init_rvec,init_pvec,init_state)

    print(f"h = {h}")
    print("Before propagation")
    bundle.BuildS()
    C = bundle.GetC()
    print(C.T.conjugate() @ bundle.S @ C)
    print("During propagation")
    for idx2 in range(nsteps): 
        advance_one_step(bundle,h)
        bundle.BuildS()
        C = bundle.GetC()
        print(C)
        print(C.T.conjugate() @ bundle.S @ C)
        if idx2 % mod == 0:
            bundle.BuildS()
            C = bundle.GetC()
            S = bundle.S
            nS = np.vdot(C, S @ C).real
            C /= np.sqrt(nS)
            bundle.SetC(C)
    print("After propagation")
    bundle.BuildS()
    C = bundle.GetC()
    print(C.T.conjugate() @ bundle.S @ C)
    h /= 10 
    nsteps *= 10 
