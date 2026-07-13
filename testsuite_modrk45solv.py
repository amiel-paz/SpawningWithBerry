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
from modrk45solv import *
import random

SEED = 987654321
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
random.seed(SEED)

nsamples = 20

ndim = 2
h=1.0e-1
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
C /= math.sqrt( np.vdot(C, S @ C) )
bundle.SetC(C)

bundle1 = copy.deepcopy(bundle)
bundle2 = copy.deepcopy(bundle)
bundle3 = copy.deepcopy(bundle)

solverk45_and_step(bundle1,h,rtol=1.0e-9,atol=1.0e-12)
print("Completed 1 stepper")

for idx in range(5):
    solverk45_and_step(bundle2,h/5,rtol=1.0e-9,atol=1.0e-12)
print("Completed 5 stepper")

for idx in range(10):
    solverk45_and_step(bundle3,h/10,rtol=1.0e-9,atol=1.0e-12)
print("Completed 10 stepper")
