from bundle import *
from models import *
from classprop import *
from testutils import *

# At crossing
ndim = 2
h=1.0e-2
nstep = 20000
model = BerryModel2DParallelTransport(0.02,3.0,15.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([0.0, 0.0])
init_pvec = np.asarray([20.0, -5.0])
init_state = 1
bundle.add_trajectory(init_rvec,init_pvec,init_state)
TBF = bundle.trajectorylist[0]
G = np.diag(np.reciprocal(TBF.get_masses()))
m = TBF.get_masses()

bundle.BuildHeff()
print(f"Quantum energy: {bundle.H[0,0]}")
Eclass = model.classical_energy(TBF.get_rvec(),TBF.get_pvec(),m,TBF.get_state())
print(f"Classical energy: {Eclass}")

E_mean, E_std = sample_phase_space_energy_MC(TBF,model,int(2.0e5))
print(f"MC Quantum energy: {E_mean}")
print(f"MC sigma: {E_std}")
E_var = projected_refined_energy_MC(TBF,model,int(5.0e1))
print(f"Projected refined Quantum energy: {E_var}")

# At crossing
ndim = 2
h=1.0e-2
nstep = 20000
model = BerryModel2DParallelTransport(0.02,3.0,15.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([0.0, 0.0])
init_pvec = np.asarray([20.0, 5.0])
init_state = 0
bundle.add_trajectory(init_rvec,init_pvec,init_state)
TBF = bundle.trajectorylist[0]
G = np.diag(np.reciprocal(TBF.get_masses()))
m = TBF.get_masses()

bundle.BuildHeff()
print(f"Quantum energy: {bundle.H[0,0]}")
Eclass = model.classical_energy(TBF.get_rvec(),TBF.get_pvec(),m,TBF.get_state())
print(f"Classical energy: {Eclass}")

E_mean, E_std = sample_phase_space_energy_MC(TBF,model,int(2.0e5))
print(f"MC Quantum energy: {E_mean}")
print(f"MC sigma: {E_std}")
E_var = projected_refined_energy_MC(TBF,model,int(5.0e1))
print(f"Projected refined Quantum energy: {E_var}")
