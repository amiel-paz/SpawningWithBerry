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

def mean_and_ci95(stack: np.ndarray):
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    std  = stack.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    half = 1.96 * std / max(n**0.5, 1.0)
    return mean, mean - half, mean + half

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
    # excited
    plt.fill_between(times, xs_lo, xs_hi, alpha=0.25, linewidth=0, label="excited 95% CI")
    plt.plot(times, xs_mean, linewidth=1.75, label="excited mean")
    # ground
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

def main():
    ap = argparse.ArgumentParser(description="Strict averaging of 2D frames + pop stats (only complete runs included).")
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

        xs_mean, xs_lo, xs_hi = mean_and_ci95(xs_stack)
        gs_mean, gs_lo, gs_hi = mean_and_ci95(gs_stack)

        np.savetxt(stats_out / "times.txt", times)
        np.savetxt(stats_out / "xs_mean.txt", xs_mean)
        np.savetxt(stats_out / "xs_lo.txt", xs_lo)
        np.savetxt(stats_out / "xs_hi.txt", xs_hi)
        np.savetxt(stats_out / "gs_mean.txt", gs_mean)
        np.savetxt(stats_out / "gs_lo.txt", gs_lo)
        np.savetxt(stats_out / "gs_hi.txt", gs_hi)

        # separate plots
        plot_with_ci(times, xs_mean, xs_lo, xs_hi,
                     f"Excited population (k={k}, mean ± 95% CI)",
                     stats_out / "pop_excited_ci.png")
        plot_with_ci(times, gs_mean, gs_lo, gs_hi,
                     f"Ground population (k={k}, mean ± 95% CI)",
                     stats_out / "pop_ground_ci.png")

        # NEW: combined plot
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

