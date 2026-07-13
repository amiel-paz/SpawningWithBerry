import numpy as np
from bundle import *
from models import *
from classprop import *
from propagate import *
from testutils import *
from multidgaussfuncs import *
from modlinsolv import *
from modcgsolv import *
from scipy.sparse.linalg import cg,minres
import scipy.linalg as la
import random

SEED = 987654321
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
random.seed(SEED)

nsamples = 10

ndim = 2
h=1.0e-2
nstep = 20000
model = BerryModel2DParallelTransport(0.02,3.0,15.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([0.0, 0.0])
init_pvec = np.asarray([20.0, 0.0])
init_state = 1
bundle.add_trajectory(init_rvec,init_pvec,init_state)
TBF = bundle.trajectorylist[0]
bundle.BuildHeff()
print(bundle.H[0,0])

bundle,E_proj = projected_refined_energy_MC(TBF, model, nsamples, rng = rng, Eclass_deviation = 0.15) 
print(E_proj)
bundle.BuildHeff()
Sevals,Sevecs = np.linalg.eigh(bundle.S)
print(Sevals)
C = bundle.GetC()
S = bundle.S
print(np.vdot(C, S @ C))

bundle.BuildHeff()
# Build your right-hand side
K = (bundle.H - 1j * bundle.SDot)
Delta = K.T.conjugate() @ S + S @ K

print( np.linalg.norm(Delta) )

X,res, _, _ = la.lstsq(S,Delta)
Ktilde = K - 0.5 * X
Delta = Ktilde.T.conjugate() @ S + S @ Ktilde
print( np.linalg.norm(Delta) )

# ================================================================
# ================================================================
from copy import deepcopy
import numpy as np
from scipy import linalg as la
from scipy.sparse.linalg import cg

# ------------------ fixed refine loop (Ri -> Rm) ------------------
max_slice = 50
hhalf = 0.5 * h                                  # <-- correct half-step

# snapshot BEFORE the classical step
init_bundle = deepcopy(bundle)
Ri = [init_bundle.trajectorylist[i].rvec.copy() for i in range(init_bundle.ntraj)]
Pi = [init_bundle.trajectorylist[i].pvec.copy() for i in range(init_bundle.ntraj)]
init_bundle.BuildHeff()
S_init = init_bundle.S
Sdot_init = init_bundle.SDot
H_init = init_bundle.H

# do the classical step (gives endpoints + midpoints)
norm_init = np.vdot(C, S @ C)
print(f"Norm before any propagation: {norm_init}\n")
info, Rf, Pf, Rm, Pm = implicit_midpoint_step_boris(bundle, h)

# metric for acceptance test = metric at the leg END (the midpoint geometry)
end_bundle = deepcopy(init_bundle)
for i in range(end_bundle.ntraj):
    end_bundle.trajectorylist[i].rvec = Rm[i]
    end_bundle.trajectorylist[i].pvec = Pm[i]
end_bundle.BuildHeff()
S_end = end_bundle.S

# Start from C0
norm_init = np.vdot(C, S @ C)
C /= math.sqrt(norm_init)
norm_init = np.vdot(C, S @ C)
print(f"Norm after normalization: {norm_init}\n")
curr_C = C.copy()

for attempt in range(1, max_slice):
    n = attempt + 1                               # number of equal subsegments
    dt_seg = hhalf / n
    test_C = C.copy()
    prev_norm = norm_init
    # march across n segments using CN at each segment MIDPOINT
    norm_conserving = True
    for k in range(n):
        t_mid = (k + 0.5) / n                     # convex parameter in [0,1]

        # convex, unit-sum interpolation in *canonical* variables
        Rmid = [(1.0 - t_mid) * Ri[i] + t_mid * Rm[i] for i in range(init_bundle.ntraj)]
        Pmid = [(1.0 - t_mid) * Pi[i] + t_mid * Pm[i] for i in range(init_bundle.ntraj)]

        # set a temporary bundle at this midpoint (no deep copies in the loop)
        temp_bundle = init_bundle                 # reuse structure, overwrite coords
        for i in range(temp_bundle.ntraj):
            temp_bundle.trajectorylist[i].rvec = Rmid[i]
            temp_bundle.trajectorylist[i].pvec = Pmid[i]

        # Build matrices at this midpoint
        temp_bundle.BuildHeff()
        S    = temp_bundle.S
        H    = temp_bundle.H
        Sdot = temp_bundle.SDot                   # already anti-Hermitian in your build
        C = temp_bundle.GetC()

        # Effective midpoint generator (your convention): Heff = H - i * tau; tau = Sdot
        H_expec = np.vdot(C, H @ C).real
        K_eff = (H - S * H_expec)  - 1j * Sdot

        # Enforce **S-Hermiticity** so Cayley in S-metric is S-unitary
        K_S, res, _, _ = la.lstsq(S, K_eff)
        test_C = PropH(K_S,test_C,dt_seg,nslice=10)

        # Norm conservation from previous step
        curr_norm = np.vdot(test_C, S_end @ test_C)
        print(f"Norm after step {k+1} of {n}: {curr_norm}\n")
        metric = abs( 1 - abs(curr_norm/prev_norm) )
        if (metric > 1.0e-6):
            print(f"Bad norm metric change of {metric} on step {k+1} of {n}. Adding more slices")
            norm_conserving = False
            break
        else:
            test_C /= math.sqrt(prev_norm)

    if not norm_conserving:
        continue

    # measure angle in the end metric S_end (phase-agnostic)
    num = np.vdot(test_C, S_end @ curr_C)
    den = np.sqrt(np.vdot(test_C, S_end @ test_C) * np.vdot(curr_C, S_end @ curr_C)) + 1e-300
    angle = np.arccos(np.clip(np.abs(num) / np.abs(den), 0.0, 1.0))
    print(f"Angle for {attempt} slices: {angle}")

    # also inspect S_end-norm to verify near-unit preservation
    norm_end = np.vdot(test_C, S_end @ test_C)
    print(f"Norm for {attempt} slices (S_end): {norm_end}\n")

    curr_C = test_C.copy()

