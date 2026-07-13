#!/usr/bin/env python3
import argparse, re, sys, subprocess, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotter import save_state_density_frame_fixed, make_fixed_grid

DIR_PAT = re.compile(r"^saved_([\d_]+)$")
EXPECTED_STEPS = list(range(0, 4980 + 1, 20))  # strict: 0..4980 by 20

def find_groups(root: Path, groups):
    by_k = {k: [] for k in groups}
    for d in root.iterdir():
        if not d.is_dir(): continue
        m = DIR_PAT.match(d.name)
        if not m: continue
        k = len([tok for tok in m.group(1).split("_") if tok])
        if k in by_k:
            by_k[k].append(d)
    for k in by_k:
        by_k[k] = sorted(by_k[k], key=lambda p: p.name)
    return by_k

def run_is_complete(d: Path):
    # require pops_* files
    for fname in ("pops_times.txt","pops_xs.txt","pops_gs.txt"):
        if not (d / fname).exists():
            return False, f"missing {fname}"
    # require all frame files for each step
    for step in EXPECTED_STEPS:
        if not (d / f"frame_{step}_GS_pop.txt").exists():
            return False, f"missing frame_{step}_GS_pop.txt"
        if not (d / f"frame_{step}_XS_pop.txt").exists():
            return False, f"missing frame_{step}_XS_pop.txt"
    # times length check
    try:
        t = np.loadtxt(d / "pops_times.txt")
        if t.ndim != 1 or len(t) != len(EXPECTED_STEPS):
            return False, f"pops_times length {len(t)} != {len(EXPECTED_STEPS)}"
    except Exception as e:
        return False, f"failed to read pops_times: {e}"
    return True, "ok"

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def mean_and_ci95_nan(stack: np.ndarray):
    """
    stack: (n_runs, T) possibly with NaNs along time.
    Returns (mean, lo, hi) using 95% CI, ignoring NaNs timewise.
    """
    # number of non-NaN samples per time
    n = np.sum(~np.isnan(stack), axis=0).astype(float)
    # protect zeros
    n_safe = np.where(n > 0, n, 1.0)
    mean = np.nanmean(stack, axis=0)
    # ddof=1 only if n>1, else 0
    std = np.nanstd(stack, axis=0, ddof=1)
    half = np.zeros_like(mean)
    mask = n > 1
    half[mask] = 1.96 * std[mask] / np.sqrt(n[mask])
    # if n<=1, CI=0
    lo = mean - half
    hi = mean + half
    return mean, lo, hi

def plot_with_ci(times, mean, lo, hi, title, outpng, label=None):
    plt.figure(figsize=(6,4), dpi=140)
    if label is None:
        plt.fill_between(times, lo, hi, alpha=0.25, linewidth=0)
        plt.plot(times, mean, linewidth=1.75)
    else:
        plt.fill_between(times, lo, hi, alpha=0.25, linewidth=0, label=f"{label} 95% CI")
        plt.plot(times, mean, linewidth=1.75, label=f"{label} mean")
    plt.xlabel("time"); plt.ylabel("population")
    plt.title(title); plt.tight_layout(); plt.savefig(outpng); plt.close()

def plot_both_with_ci(times,
                      xs_mean, xs_lo, xs_hi,
                      gs_mean, gs_lo, gs_hi,
                      title, outpng):
    plt.figure(figsize=(6,4), dpi=140)
    plt.fill_between(times, xs_lo, xs_hi, alpha=0.25, linewidth=0, label="excited 95% CI")
    plt.plot(times, xs_mean, linewidth=1.75, label="excited mean")
    plt.fill_between(times, gs_lo, gs_hi, alpha=0.25, linewidth=0, label="ground 95% CI")
    plt.plot(times, gs_mean, linewidth=1.75, label="ground mean")
    plt.xlabel("time"); plt.ylabel("population")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()

def try_make_movie(glob_pat: str, out_mp4: Path, fps: int, qv: int):
    if shutil.which("ffmpeg") is None:
        print(f"[note] ffmpeg not found; skipped {out_mp4}", file=sys.stderr)
        return
    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error",
           "-framerate",str(fps),"-pattern_type","glob","-i",glob_pat,
           "-vf","format=yuv420p","-c:v","mpeg4","-q:v",str(qv),str(out_mp4)]
    subprocess.run(cmd, check=True)

def centroids_and_momentum_for_run(run_dir: Path, X, Y, times, massx, massy, pop_min):
    """
    Returns dict with keys:
      xs_px, xs_py, gs_px, gs_py   each shape (T,)
    NaNs when population < pop_min (to avoid noise).
    """
    T = len(times)
    xs_x = np.full(T, np.nan); xs_y = np.full(T, np.nan)
    gs_x = np.full(T, np.nan); gs_y = np.full(T, np.nan)

    # precompute denominators/pop and centroids
    for i, step in enumerate(EXPECTED_STEPS):
        rho_gs = np.loadtxt(run_dir / f"frame_{step}_GS_pop.txt")  # already probability mass per pixel
        rho_xs = np.loadtxt(run_dir / f"frame_{step}_XS_pop.txt")
        pop_gs = float(np.sum(rho_gs))
        pop_xs = float(np.sum(rho_xs))

        if pop_gs > pop_min:
            gs_x[i] = float((rho_gs * X).sum() / pop_gs)
            gs_y[i] = float((rho_gs * Y).sum() / pop_gs)
        if pop_xs > pop_min:
            xs_x[i] = float((rho_xs * X).sum() / pop_xs)
            xs_y[i] = float((rho_xs * Y).sum() / pop_xs)

    # finite-difference velocities (central; fwd/backward at edges)
    def vel_from_centroid(c, t):
        v = np.full_like(c, np.nan, dtype=float)
        dt = np.diff(t)
        if len(t) >= 3:
            # central differences where we have neighbors and non-NaN
            for i in range(1, len(t)-1):
                if np.isfinite(c[i-1]) and np.isfinite(c[i+1]):
                    dtc = t[i+1] - t[i-1]
                    if dtc != 0:
                        v[i] = (c[i+1] - c[i-1]) / dtc
        # forward/backward if possible
        if np.isfinite(c[0]) and np.isfinite(c[1]) and dt[0] != 0:
            v[0] = (c[1] - c[0]) / dt[0]
        if np.isfinite(c[-1]) and np.isfinite(c[-2]) and dt[-1] != 0:
            v[-1] = (c[-1] - c[-2]) / dt[-1]
        return v

    xs_vx = vel_from_centroid(xs_x, times)
    xs_vy = vel_from_centroid(xs_y, times)
    gs_vx = vel_from_centroid(gs_x, times)
    gs_vy = vel_from_centroid(gs_y, times)

    # mechanical momentum components: π = m * v
    xs_px = massx * xs_vx
    xs_py = massy * xs_vy
    gs_px = massx * gs_vx
    gs_py = massy * gs_vy

    return dict(xs_px=xs_px, xs_py=xs_py, gs_px=gs_px, gs_py=gs_py)

def plot_ci_series(times, mean, lo, hi, ylabel, title, outpng):
    plt.figure(figsize=(6,4), dpi=140)
    plt.fill_between(times, lo, hi, alpha=0.25, linewidth=0)
    plt.plot(times, mean, linewidth=1.75)
    plt.xlabel("time"); plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Strict averaging of 2D frames + pop & momentum stats (only complete runs included).")
    ap.add_argument("--root", type=str, default=".", help="where saved_* folders live")
    ap.add_argument("--groups", type=int, nargs="+", default=[1,4,8])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--qv", type=int, default=3)          # mpeg4 quality (1 best)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--Lx", type=float, default=20.0)
    ap.add_argument("--Ly", type=float, default=20.0)
    ap.add_argument("--vmax", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="averaged_outputs")
    ap.add_argument("--movies", action="store_true")
    ap.add_argument("--massx", type=float, default=1.0, help="mass in x for mechanical momentum")
    ap.add_argument("--massy", type=float, default=1.0, help="mass in y for mechanical momentum")
    ap.add_argument("--pop_min", type=float, default=1e-9, help="min pop to compute centroid; else NaN")
    args = ap.parse_args()

    root = Path(args.root)
    groups = find_groups(root, set(args.groups))
    print("Found groups:", {k: len(v) for k, v in groups.items()})

    X, Y = make_fixed_grid(nx=args.nx, ny=args.ny, Lx=args.Lx, Ly=args.Ly)

    outroot = Path(args.out); ensure_dir(outroot)

    for k, dirs in groups.items():
        if not dirs:
            print(f"[k={k}] no dirs")
            continue

        kept = []
        for d in dirs:
            ok, why = run_is_complete(d)
            if ok: kept.append(d)
            else:  print(f"[k={k}] skipping {d.name}: {why}")

        if not kept:
            print(f"[k={k}] no complete runs; skipping")
            continue

        print(f"[k={k}] averaging {len(kept)} complete run(s)")
        grp_out   = outroot / f"k{k}"
        frames_out= grp_out / "frames_adiabatic_avg"
        stats_out = grp_out / "stats"
        movies_out= grp_out / "movies"
        ensure_dir(frames_out); ensure_dir(stats_out); ensure_dir(movies_out)

        # ---------- populations ----------
        times = np.loadtxt(kept[0] / "pops_times.txt")
        xs_stack, gs_stack = [], []
        for d in kept:
            xs_stack.append(np.loadtxt(d / "pops_xs.txt"))
            gs_stack.append(np.loadtxt(d / "pops_gs.txt"))
        xs_stack = np.vstack(xs_stack); gs_stack = np.vstack(gs_stack)

        from math import isfinite
        xs_mean, xs_lo, xs_hi = mean_and_ci95_nan(xs_stack)
        gs_mean, gs_lo, gs_hi = mean_and_ci95_nan(gs_stack)

        np.savetxt(stats_out / "times.txt", times)
        np.savetxt(stats_out / "xs_mean.txt", xs_mean)
        np.savetxt(stats_out / "xs_lo.txt", xs_lo)
        np.savetxt(stats_out / "xs_hi.txt", xs_hi)
        np.savetxt(stats_out / "gs_mean.txt", gs_mean)
        np.savetxt(stats_out / "gs_lo.txt", gs_lo)
        np.savetxt(stats_out / "gs_hi.txt", gs_hi)

        plot_with_ci(times, xs_mean, xs_lo, xs_hi,
                     f"Excited population (k={k}, mean ± 95% CI)",
                     stats_out / "pop_excited_ci.png")
        plot_with_ci(times, gs_mean, gs_lo, gs_hi,
                     f"Ground population (k={k}, mean ± 95% CI)",
                     stats_out / "pop_ground_ci.png")

        plot_both_with_ci(times,
                          xs_mean, xs_lo, xs_hi,
                          gs_mean, gs_lo, gs_hi,
                          title=f"Populations (k={k}, mean ± 95% CI)",
                          outpng=stats_out / "pop_both_ci.png")

        # ---------- averaged frames ----------
        for step in EXPECTED_STEPS:
            gs_acc = xs_acc = None
            for d in kept:
                gs = np.loadtxt(d / f"frame_{step}_GS_pop.txt")
                xs = np.loadtxt(d / f"frame_{step}_XS_pop.txt")
                gs_acc = gs if gs_acc is None else (gs_acc + gs)
                xs_acc = xs if xs_acc is None else (xs_acc + xs)
            gs_avg = gs_acc / len(kept)
            xs_avg = xs_acc / len(kept)

            save_state_density_frame_fixed(
                frames_out, step, X, Y, gs_avg,
                state_index=0,
                title_prefix=f"Averaged adiabatic density (k={k})",
                fixed_vmax=args.vmax,
                use_prob_mass=False
            )
            save_state_density_frame_fixed(
                frames_out, step, X, Y, xs_avg,
                state_index=1,
                title_prefix=f"Averaged adiabatic density (k={k})",
                fixed_vmax=args.vmax,
                use_prob_mass=False
            )

        # ---------- mechanical momentum from density (centroid finite-diff) ----------
        xs_px_runs, xs_py_runs, gs_px_runs, gs_py_runs = [], [], [], []
        for d in kept:
            mom = centroids_and_momentum_for_run(
                d, X, Y, times,
                massx=args.massx, massy=args.massy,
                pop_min=args.pop_min
            )
            xs_px_runs.append(mom["xs_px"])
            xs_py_runs.append(mom["xs_py"])
            gs_px_runs.append(mom["gs_px"])
            gs_py_runs.append(mom["gs_py"])

        xs_px_stack = np.vstack(xs_px_runs)
        xs_py_stack = np.vstack(xs_py_runs)
        gs_px_stack = np.vstack(gs_px_runs)
        gs_py_stack = np.vstack(gs_py_runs)

        xs_px_mean, xs_px_lo, xs_px_hi = mean_and_ci95_nan(xs_px_stack)
        xs_py_mean, xs_py_lo, xs_py_hi = mean_and_ci95_nan(xs_py_stack)
        gs_px_mean, gs_px_lo, gs_px_hi = mean_and_ci95_nan(gs_px_stack)
        gs_py_mean, gs_py_lo, gs_py_hi = mean_and_ci95_nan(gs_py_stack)

        # Save series
        np.savetxt(stats_out / "xs_px_mean.txt", xs_px_mean); np.savetxt(stats_out / "xs_px_lo.txt", xs_px_lo); np.savetxt(stats_out / "xs_px_hi.txt", xs_px_hi)
        np.savetxt(stats_out / "xs_py_mean.txt", xs_py_mean); np.savetxt(stats_out / "xs_py_lo.txt", xs_py_lo); np.savetxt(stats_out / "xs_py_hi.txt", xs_py_hi)
        np.savetxt(stats_out / "gs_px_mean.txt", gs_px_mean); np.savetxt(stats_out / "gs_px_lo.txt", gs_px_lo); np.savetxt(stats_out / "gs_px_hi.txt", gs_px_hi)
        np.savetxt(stats_out / "gs_py_mean.txt", gs_py_mean); np.savetxt(stats_out / "gs_py_lo.txt", gs_py_lo); np.savetxt(stats_out / "gs_py_hi.txt", gs_py_hi)

        # Plots
        plot_ci_series(times, xs_px_mean, xs_px_lo, xs_px_hi,
                       ylabel="πx (mechanical)", title=f"Excited πx (k={k})", outpng=stats_out / "px_excited_ci.png")
        plot_ci_series(times, xs_py_mean, xs_py_lo, xs_py_hi,
                       ylabel="πy (mechanical)", title=f"Excited πy (k={k})", outpng=stats_out / "py_excited_ci.png")
        plot_ci_series(times, gs_px_mean, gs_px_lo, gs_px_hi,
                       ylabel="πx (mechanical)", title=f"Ground πx (k={k})", outpng=stats_out / "px_ground_ci.png")
        plot_ci_series(times, gs_py_mean, gs_py_lo, gs_py_hi,
                       ylabel="πy (mechanical)", title=f"Ground πy (k={k})", outpng=stats_out / "py_ground_ci.png")

        # ---------- movies ----------
        if args.movies:
            gs_glob = (frames_out / "frame_ground_*.png").as_posix()
            xs_glob = (frames_out / "frame_excited_*.png").as_posix()
            try:
                try_make_movie(gs_glob, movies_out / f"adiabatic_k{k}_ground.mp4", args.fps, args.qv)
                try_make_movie(xs_glob, movies_out / f"adiabatic_k{k}_excited.mp4", args.fps, args.qv)
                if shutil.which("ffmpeg"):
                    subprocess.run([
                        "ffmpeg","-y","-hide_banner","-loglevel","error",
                        "-framerate",str(args.fps),"-pattern_type","glob","-i",xs_glob,
                        "-framerate",str(args.fps),"-pattern_type","glob","-i",gs_glob,
                        "-filter_complex","[0:v][1:v]hstack=inputs=2,format=yuv420p",
                        "-c:v","mpeg4","-q:v",str(args.qv),
                        str(movies_out / f"adiabatic_k{k}_excited_vs_ground.mp4")
                    ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[warn] ffmpeg failed (k={k}): {e}", file=sys.stderr)

    print("Done.")

if __name__ == "__main__":
    main()

