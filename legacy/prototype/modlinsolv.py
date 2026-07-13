import numpy as np
from scipy.sparse import issparse, diags
from scipy.sparse.linalg import (
    LinearOperator, lsqr, cg, minres, gmres, onenormest
)

# ---------- helpers ----------
def _as_linear_op(A, like_vec):
    """
    Wrap A as a LinearOperator with BOTH matvec and rmatvec.
    - dense: uses A @ v and A^H @ v
    - sparse: uses A @ v and A.H @ v
    - LinearOperator: must already provide rmatvec (LSQR needs it)
    """
    if isinstance(A, LinearOperator):
        if getattr(A, "rmatvec", None) is None:
            raise NotImplementedError("Provided LinearOperator lacks rmatvec; LSQR needs it.")
        return A

    # Determine dtype to accommodate complex inputs
    import numpy as np
    target_dtype = np.result_type(getattr(A, "dtype", np.float64), like_vec.dtype)

    if issparse(A):
        shape = A.shape
        def mv(v):   return A @ v
        def rmv(v):  return (A.conj().T) @ v
        return LinearOperator(shape, matvec=mv, rmatvec=rmv, dtype=target_dtype)

    # Dense / array-like
    A_arr = np.asarray(A)
    shape = A_arr.shape
    def mv(v):   return A_arr @ v
    def rmv(v):  return (A_arr.conj().T) @ v
    return LinearOperator(shape, matvec=mv, rmatvec=rmv, dtype=target_dtype)

def _jacobi_prec(A):
    """Jacobi preconditioner M ≈ A^{-1} using diag(A). Safe when zeros on diag."""
    # Try to get diagonal cheaply
    if issparse(A):
        d = A.diagonal()
    elif isinstance(A, LinearOperator):
        # Fallback probe for diagonal (cost: n matvecs)
        n = A.shape[0]
        d = np.empty(n, dtype=np.complex128)
        e = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            e[i] = 1.0
            d[i] = (A @ e)[i]
            e[i] = 0.0
    else:
        A = np.asarray(A)
        d = np.diag(A)
    d = np.asarray(d, dtype=np.complex128)
    d = np.where(np.abs(d) > 0, d, 1.0 + 0j)
    invd = 1.0 / d
    n = d.size
    return LinearOperator((n, n), matvec=lambda v: invd * v, dtype=invd.dtype)

def _backward_error(Aop, x, r, Anorm1=None):
    """Normwise backward error: ||r - A x|| / (||A||*||x|| + ||r||). Use 1-norm."""
    Ax = Aop @ x
    res = r - Ax
    num = np.linalg.norm(res, 1)
    if Anorm1 is None:
        try:
            Anorm1 = float(onenormest(Aop))
        except Exception:
            # crude fallback
            Anorm1 = 1.0
    denom = Anorm1 * np.linalg.norm(x, 1) + np.linalg.norm(r, 1)
    if denom == 0:
        return 0.0
    return float(num / denom)

# ---------- main autopilot ----------
def solve_Sx_eq_r_autopilot(
    S, r,
    *,
    assume_hermitian=True,          # True for overlap S, or Hermitian step operator
    assume_psd=True,                # True for plain S; False if operator may be indefinite
    ls_tol=1e-12,                   # LSQR atol/btol
    cg_rtol=1e-12,                  # CG/MINRES relative tolerance
    maxiter=None,
    use_precond=True,
    prefer_minres_if_hermitian_indef=True,
):
    """
    Adaptive solve for S x = r (complex OK), with CERFACS-style checks.

    Steps:
      1) LSQR to test consistency & get x_ls (least-squares minimizer).
      2) If ||S x_ls - r|| / ||r|| small => consistent; refine with:
         - CG if Hermitian PSD;
         - MINRES if Hermitian indefinite;
         - otherwise GMRES.
      3) Return x plus metadata including residuals and backward-error history.
    """
    r = np.asarray(r)
    Aop = _as_linear_op(S, r)
    n = Aop.shape[0]

    # 1) LS gate
    ls = lsqr(Aop, r, atol=ls_tol, btol=ls_tol, iter_lim=maxiter)
    x_ls = ls[0].astype(np.complex128, copy=False)
    r_proj = Aop @ x_ls
    resid = r - r_proj
    resid_norm = float(np.linalg.norm(resid))
    r_norm = float(np.linalg.norm(r)) or 1.0
    rel_resid = resid_norm / r_norm

    # Consistency decision
    consistent = (rel_resid <= 10.0 * ls_tol)

    # Preconditioner (safe default)
    M = _jacobi_prec(S) if use_precond else None

    # Backward-error estimate uses 1-norm of A
    try:
        A1 = float(onenormest(Aop))
    except Exception:
        A1 = None

    history = {
        "iter": [],
        "true_resid_norm": [],
        "rel_true_resid": [],
        "backward_error": [],
    }

    def make_callback(name):
        # Capture values per iteration
        def _cb(xk):
            Axk = Aop @ xk
            rk = r - Axk
            rn = float(np.linalg.norm(rk))
            rel = rn / (r_norm if r_norm else 1.0)
            be = _backward_error(Aop, xk, r, A1)
            history["iter"].append(len(history["iter"]) + 1)
            history["true_resid_norm"].append(rn)
            history["rel_true_resid"].append(rel)
            history["backward_error"].append(be)
        return _cb

    meta = {
        "mode": "least_squares",
        "lsqr": {"istop": ls[1], "itn": ls[2]},
        "ls_rel_resid": rel_resid,
        "r_projected": r_proj,
        "history": history,
    }

    if not consistent:
        # Inconsistent: LSQR solution is the projection; we're done.
        return x_ls, meta

    # 2) Consistent: refine with appropriate Krylov solver
    x0 = x_ls
    cb = make_callback("refine")

    if assume_hermitian:
        if assume_psd:
            # CG (Hermitian SPD/PSD)
            x_cg, info = cg(Aop, r, x0=x0, M=M, rtol=cg_rtol, maxiter=maxiter, callback=cb)
            meta["mode"] = "exact_cg"
            meta["cg_info"] = {"info": info}
            return x_cg, meta
        else:
            # Hermitian but possibly indefinite → MINRES
            x_mr, info = minres(Aop, r, x0=x0, M=M, rtol=cg_rtol, maxiter=maxiter, callback=cb)
            meta["mode"] = "exact_minres"
            meta["minres_info"] = {"info": info}
            return x_mr, meta
    else:
        # Non-Hermitian → GMRES
        x_gr, info = gmres(Aop, r, x0=x0, M=M, tol=cg_rtol, restart=None, maxiter=maxiter, callback=cb)
        meta["mode"] = "exact_gmres"
        meta["gmres_info"] = {"info": info}
        return x_gr, meta

