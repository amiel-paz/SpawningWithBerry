from bundle import *
from models import *
from classprop import *
from testutils import *
import cmath
import numpy as np
from propagate import *

# At crossing
ndim = 2
model = BerryModel2DParallelTransport(0.02,3.0,15.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([0.0, 0.0])
init_pvec = np.asarray([20.0, 0.0])
init_state = 1
bundle.add_trajectory(init_rvec,init_pvec,init_state)
init_state = 0
bundle.add_trajectory(init_rvec,init_pvec,init_state)
TBF = bundle.trajectorylist[0]
G = np.diag(np.reciprocal(TBF.get_masses()))
m = TBF.get_masses()

bundle.BuildHeff()
print(f"Quantum coupling: {bundle.H[0,1]}")
d_vec = model.d(init_rvec,1,0)
coup = np.dot(d_vec,init_pvec/m)
coup += model.D(init_rvec,m,1,0)
print(f"Infinitely localized coupling: {coup}")

curr_eps = 1
for idx in range(10):
    H = bundle.BuildPE_eps(curr_eps) + bundle.BuildKE_eps(curr_eps)
    print(f"Scaled Quantum coupling ({curr_eps}): {H[0,1]}")
    curr_eps /= 10

# At crossing with displaced geoms
ndim = 2
model = BerryModel2DParallelTransport(0.02,3.0,15.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([0.0, 0.0])
init_pvec = np.asarray([20.0, 0.0])
init_state = 1
bundle.add_trajectory(init_rvec,init_pvec,init_state)
init_rvec = np.asarray([-0.2, 0.0])
init_pvec = np.asarray([20.0, 0.0])
init_state = 1
bundle.add_trajectory(init_rvec,init_pvec,init_state)
init_rvec = np.asarray([0.2, 0.0])
init_pvec = np.asarray([20.0, 0.0])
init_state = 1
bundle.add_trajectory(init_rvec,init_pvec,init_state)

TBF = bundle.trajectorylist[0]
G = np.diag(np.reciprocal(TBF.get_masses()))
m = TBF.get_masses()
bundle.BuildHeff()
Sdot_full = bundle.SDot.conj().T + bundle.SDot
Rphys = 1j * (bundle.Heff.conj().T @ bundle.S - bundle.S @ bundle.Heff) + Sdot_full
Rphys_err = np.linalg.norm(Rphys)
print("|| TDVP unitarity residual || =", Rphys_err)
print("||H^† - H|| =", np.linalg.norm(bundle.H.conj().T - bundle.H))
