# --- Headless plotting (no windows) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from pathlib import Path
import numpy as np
import math

# ---------------- Fixed grid to match your split-operator script ----------------
def make_fixed_grid(nx=256, ny=256, Lx=20.0, Ly=20.0):
    """
    Same lattice as your FFT script:
      x = linspace(-Lx/2, Lx/2 - dx, nx) with dx=Lx/nx (periodic-friendly)
    """
    dx, dy = Lx / nx, Ly / ny
    x = np.linspace(-Lx/2,  Lx/2 - dx, nx)
    y = np.linspace(-Ly/2,  Ly/2 - dy, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    return X, Y

def _grid_spacing(X, Y):
    dx = (X.max() - X.min()) / X.shape[1]
    dy = (Y.max() - Y.min()) / Y.shape[0]
    return dx, dy

# ---------------- Heller FG utilities (ħ=1, no gamma) ----------------
def _fg2d_on_grid(R, P, widths, X, Y):
    """
    χ_k(x,y) = N * exp[ -a_x (x-Rx)^2 - a_y (y-Ry)^2 + i * P·(r-R) ].
    Accepts scalar widths or 2-vector (ax, ay). Uses Re(widths) for normalization.
    """
    Rx, Ry = float(R[0]), float(R[1])
    Px, Py = float(P[0]), float(P[1])

    # allow scalar or per-dim
    if np.ndim(widths) == 0:
        ax = ay = complex(widths)
    else:
        ax, ay = complex(widths[0]), complex(widths[1])

    axR, ayR = np.real(ax), np.real(ay)
    if axR <= 0 or ayR <= 0:
        raise ValueError(f"Re(widths) must be > 0; got Re(ax)={axR}, Re(ay)={ayR}")

    # Normalization for separable product of 1D Gaussians
    N = (2.0 * axR / np.pi) ** 0.25 * (2.0 * ayR / np.pi) ** 0.25

    dx = X - Rx
    dy = Y - Ry
    return N * np.exp(-ax * dx * dx - ay * dy * dy + 1j * (Px * dx + Py * dy))

def reconstruct_state_on_grid(bundle, X, Y, state_index):
    """
    ψ_state(x,y) = sum_{k with traj.state == state_index} C_k χ_k(x,y)
    Assumes bundle.GetC() aligns with bundle.trajectorylist order.
    """
    C = np.asarray(bundle.GetC(), dtype=np.complex128)
    psi = np.zeros_like(X, dtype=np.complex128)

    for k, traj in enumerate(bundle.trajectorylist):
        if traj.get_state() != state_index:
            continue
        chi = _fg2d_on_grid(traj.rvec, traj.pvec, traj.widths, X, Y)
        psi += C[k] * chi
    return psi

# ---------------- Save a fixed-style frame (no autoscaling) ----------------
def save_state_density_frame_fixed(outdir, step, X, Y, rho_state,
                                   state_index=0,
                                   title_prefix="Adiabatic state density",
                                   fixed_vmax=1e-3,
                                   use_prob_mass=False):
    """
    If use_prob_mass=True, multiply |psi|^2 by dx*dy so the colormap reflects
    discrete probability mass per pixel (handy for comparing grids).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extent = [X.min(), X.max(), Y.min(), Y.max()]  # fixed viewport

    Z = np.real(rho_state)
    if use_prob_mass:
        dx, dy = _grid_spacing(X, Y)
        Z = Z * dx * dy

    # Friendly label
    state_name = {0: "ground", 1: "excited"}.get(int(state_index), f"state{state_index}")

    plt.figure(figsize=(5, 4), dpi=140)
    plt.imshow(Z, origin='lower', extent=extent, aspect='auto',
               vmin=0.0, vmax=fixed_vmax)
    plt.colorbar()
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f"{title_prefix} ({state_name})  step={step}")
    plt.tight_layout()
    plt.savefig(outdir / f'frame_{state_name}_{step:06d}.png')
    plt.close()

