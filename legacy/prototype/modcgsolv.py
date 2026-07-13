import numpy as np
from scipy.sparse.linalg import cg

def cg_overlap_solve(S, r, tol=1e-12, maxiter=None, rel_cut=1e-10):
    """
    Solve S x = r for Hermitian (semi)definite S.
    Auto-projects onto range(S) so null/near-null directions are frozen.

    Parameters
    ----------
    S : (n,n) ndarray (complex or real Hermitian)
    r : (n,)   ndarray (same dtype as S)
    tol : float         CG relative tolerance
    maxiter : int|None  CG max iterations (defaults to 2n internally)
    rel_cut : float     relative eigen cutoff (keep eigs > rel_cut * max_eig)

    Returns
    -------
    x : (n,) ndarray    solution with null-space part = 0 (frozen)
    info : dict         {'iter', 'converged', 'res_norm', 'rank_kept'}
    """
    S = np.asarray(S)
    r = np.asarray(r, dtype=np.result_type(S.dtype, r.dtype))

    # 1) Eigenthreshold to get a basis for range(S)
    w, U = np.linalg.eigh(S)                 # Hermitian eigendecomp
    lam_max = float(w.max(initial=1.0))
    keep = w > rel_cut * lam_max             # drop tiny/zero modes
    if not np.any(keep):
        # S is (numerically) zero ⇒ best we can do is x=0
        return np.zeros_like(r), {"iter": 0, "converged": True, "res_norm": float(np.linalg.norm(r)), "rank_kept": 0}

    Ur = U[:, keep]                          # columns spanning range(S)

    # 2) Project RHS and form reduced SPD system
    br = Ur.conj().T @ r                     # = Ur^H r
    Sr = Ur.conj().T @ (S @ Ur)              # SPD in reduced space

    # 3) CG on reduced system (tiny and well-conditioned)
    y0 = None
    y, info_cg = cg(Sr, br, x0=y0, rtol=tol, maxiter=maxiter)
    x = Ur @ y                               # lift back

    # 4) Diagnostics
    res = r - S @ x
    info = {
        "iter": info_cg if isinstance(info_cg, int) else -1,
        "converged": (info_cg == 0),
        "res_norm": float(np.linalg.norm(res)),
        "rank_kept": int(keep.sum()),
    }
    return x, info

