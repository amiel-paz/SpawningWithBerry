import numpy as np
import copy
from bundle import Bundle

def _rebuild_bundle_keep_indices(bundle, keep_idx):
    """Rebuild a new Bundle with only trajectories in keep_idx, preserving order."""
    model = bundle.model
    ndim  = model.ndim
    newB  = Bundle(ndim, model)
    for k in keep_idx:
        t = bundle.trajectorylist[k]
        newB.add_trajectory(t.get_rvec().copy(), t.get_pvec().copy(), t.get_state())
    # map coefficients (assume old C aligned with trajectorylist order)
    oldC = bundle.GetC()
    newC = np.zeros(len(keep_idx), dtype=np.complex128)
    for j, k in enumerate(keep_idx):
        newC[j] = oldC[k]
    newB.SetC(newC)
    newB.BuildS()
    return newB

def prune_children_by_singularity(
    bundle, 
    child_state, 
    coup_mags_for_children,  # 1D array aligned with the order you added children
    eig_rel_thresh=1e-8,     # relative eigen cutoff for "bad" modes
    max_cond=1e10,           # acceptable condition number cap
    verbose=True
):
    """
    Greedy prune of child-state TBFs:
      - Build child Gram S_cc.
      - Find tiny-eigenvalue modes of S_cc.
      - Score each child j by its 'redundancy score':
            R_j = sum_{bad modes m} w_m * |U[j,m]|^2
        where w_m weights smaller eigenvalues higher.
      - Iteratively drop the child with largest R_j; on ties, drop the one with
        *smaller* coup_mags (we prefer to keep larger coupling).
      - Stop when cond(S_cc) is acceptable or no bad modes remain.
    NOTE: We assume you added the parent first (index 0), then children in the
          exact order of coup_mags_for_children.
    """
    bundle.BuildS()
    S = 0.5*(bundle.S + bundle.S.conj().T)

    # indices: parent assumed at 0; children follow in insertion order
    child_idx = [i for i, t in enumerate(bundle.trajectorylist) if t.state == child_state]
    if len(child_idx) == 0:
        if verbose:
            print("[prune] no children to prune.")
        return bundle, child_idx

    # Sanity: map coup_mags to these children (assumes you added only child_state in the loop)
    if len(coup_mags_for_children) != len(child_idx):
        if verbose:
            print(f"[prune][warn] coup_mags length {len(coup_mags_for_children)} "
                  f"!= #children {len(child_idx)}. Will truncate to min length.")
    L = min(len(coup_mags_for_children), len(child_idx))
    child_idx = child_idx[:L]
    coup_mags = np.asarray(coup_mags_for_children[:L], float)

    # Work on a mutable list of survivors
    keep_children = child_idx.copy()

    def child_block_cond_and_eigs(keep):
        Scc = S[np.ix_(keep, keep)]
        # Hermitian safety
        Scc = 0.5*(Scc + Scc.conj().T)
        # eigendecomp
        lam, U = np.linalg.eigh(Scc)
        lam = np.maximum(lam, 0.0)  # PSD guard
        lam_max = lam.max() if lam.size else 0.0
        cond = (lam_max / lam.min()) if (lam.size and lam.min() > 0.0) else np.inf
        # bad modes: very small eigenvalues
        lam_thresh = eig_rel_thresh * max(lam_max, 1.0)
        bad_mask = lam <= lam_thresh
        return cond, lam, U, bad_mask, lam_thresh

    # Main prune loop
    while True:
        cond, lam, U, bad_mask, lam_thresh = child_block_cond_and_eigs(keep_children)

        if verbose:
            print(f"[prune] child count={len(keep_children)}, cond(S_cc)≈{cond:.3e}, "
                  f"#bad={int(bad_mask.sum())}, lam_min={lam.min() if lam.size else np.nan:.3e}")

        # Stop if well-conditioned or nothing to prune
        if (cond <= max_cond) or (bad_mask.sum() == 0) or (len(keep_children) <= 1):
            break

        # weights: smaller eigenvalues -> larger weight
        eps = 1e-30
        lam_bad = lam[bad_mask]
        w = (lam_thresh / (lam_bad + eps))  # emphasize the worst tiny modes

        # Redundancy score for each child column: R_j = Σ_m w_m |U[j,m]|^2
        U_bad = U[:, bad_mask]  # shape (nchild, nbad)
        R = (np.abs(U_bad)**2 @ w)

        # Choose child to drop:
        #   highest R_j first; break ties by smaller coup_mags
        # Map R to the *global* child order to compare coup_mags aligned with added order
        # Build a list of (global_idx, R_j, coup_j)
        trip = []
        for local_j, gidx in enumerate(keep_children):
            # local_j maps into R; global child ordinal is its position in child_idx
            try:
                child_ord = child_idx.index(gidx)
                coup = coup_mags[child_ord]
            except ValueError:
                coup = 0.0
            trip.append((gidx, R[local_j], coup))

        # sort: R desc (drop worst), coup asc (prefer to keep larger coupling)
        trip.sort(key=lambda t: (-t[1], t[2]))
        drop_gidx, drop_R, drop_coup = trip[0]

        if verbose:
            print(f"[prune] dropping child idx={drop_gidx}  (R={drop_R:.3e}, coup={drop_coup:.3e})")

        # remove it
        keep_children.remove(drop_gidx)

    # Rebuild bundle with parent + kept children (and any other non-child-state TBFs)
    keep_all = []
    for i, t in enumerate(bundle.trajectorylist):
        if t.state != child_state:
            keep_all.append(i)
    keep_all += keep_children
    keep_all = sorted(keep_all)

    pruned = _rebuild_bundle_keep_indices(bundle, keep_all)
    if verbose:
        print(f"[prune] kept {len(keep_children)} children (from {len(child_idx)}). "
              f"New ntraj={pruned.ntraj}")
    return pruned, keep_children

def select_diverse_by_time_and_geometry(child_rvecs, child_pvecs, coup_mags, saved_steps,
                                        K_per_bucket=3, bucket_width=25,
                                        r_tol=0.15, p_tol=0.5):
    """
    Return indices of a subset that:
      - keeps top-K per time bucket by coupling
      - ensures geometric diversity per bucket (no near-duplicates in (R,P))
    """
    child_rvecs = np.asarray(child_rvecs, dtype=np.complex128).real
    child_pvecs = np.asarray(child_pvecs, dtype=np.complex128).real
    coup_mags   = np.asarray(coup_mags, float)
    saved_steps = np.asarray(saved_steps, float).real.astype(int)

    sel = []
    if len(saved_steps) == 0:
        return np.array(sel, int)

    smin, smax = saved_steps.min(), saved_steps.max()
    for s0 in range(smin, smax+1, bucket_width):
        mask = (saved_steps >= s0) & (saved_steps < s0 + bucket_width)
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            continue
        # sort by coupling desc
        idxs = idxs[np.argsort(-coup_mags[idxs])]
        keep_bucket = []
        for j in idxs:
            rj = child_rvecs[j]; pj = child_pvecs[j]
            if not keep_bucket:
                keep_bucket.append(j)
            else:
                too_close = False
                for k in keep_bucket:
                    rk = child_rvecs[k]; pk = child_pvecs[k]
                    if np.linalg.norm(rj - rk) < r_tol and np.linalg.norm(pj - pk) < p_tol:
                        too_close = True; break
                if not too_close:
                    keep_bucket.append(j)
            if len(keep_bucket) >= K_per_bucket:
                break
        sel.extend(keep_bucket)

    return np.array(sorted(sel), dtype=int)

