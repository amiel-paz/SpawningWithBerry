#!/usr/bin/env python3
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def read_first_pair(path: Path):
    with open(path, "r") as f:
        first = f.readline().strip()
    # Grab only the first two tokens; ignore anything extra
    parts = first.split()
    if len(parts) < 2:
        raise ValueError(f"{path} first line has fewer than 2 numbers: {first!r}")
    try:
        x, y = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"{path} first line not parseable as floats: {first!r}")
    return x, y

def main():
    files = sorted(Path(p) for p in glob.glob("saved_*/parent_pvecs.txt"))
    if not files:
        raise SystemExit("No files matched saved_*/parent_pvecs.txt")

    xs, ys = [], []
    bad = []
    for p in files:
        try:
            x, y = read_first_pair(p)
            xs.append(x); ys.append(y)
        except Exception as e:
            bad.append((p, str(e)))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(xs, ys, s=20, alpha=0.35, edgecolor="none")  # slightly transparent points

    # σ and 2σ circles (Gaussian width = 1 in both dims), centered at (-3, 0)
    cx, cy = 20.0, 0.0
    sigma_P = 1.0
    for r, lw, label in [(1.0 * sigma_P, 1.5, r"$\sigma$"), (2.0 * sigma_P, 1.5, r"$2\sigma$")]:
        c = Circle((cx, cy), r, fill=False, linewidth=lw)
        ax.add_patch(c)
        ax.text(cx + r * 1.02, cy, label, va="center", ha="left", fontsize=10)

    # Aesthetics
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Parent pvecs: scatter with σ and 2σ (μ = (-3, 0), σ = 1)")
    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_aspect("equal")  # circles look like circles

    ax.set_xlim(cx - 3.0, cx + 3.0)
    ax.set_ylim(cy - 3.0, cy + 3.0)

    out = Path("parent_pvecs_scatter.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out.resolve()}")

    if bad:
        print("\nSkipped files due to parse issues:")
        for p, msg in bad:
            print(f"  - {p}: {msg}")

if __name__ == "__main__":
    main()

