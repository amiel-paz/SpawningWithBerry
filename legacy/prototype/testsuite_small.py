from bundle import *
from models import *
from classprop import *
from testutils import *

state=0

print("Away from crossing")
ndim = 2
h=1.0e-2
nstep = 20000
model = BerryModel2DParallelTransport(0.02,3.0,15.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([-3.0, 0.0])
init_pvec = np.asarray([20.0, 0.0])
init_state = state
bundle.add_trajectory(init_rvec,init_pvec,init_state)
TBF = bundle.trajectorylist[0]
G = np.diag(np.reciprocal(TBF.get_masses()))
m = TBF.get_masses()

bundle.BuildHeff()
print(f"Quantum energy: {bundle.H[0,0]}")
print(f"Closer to exact Quantum energy: {model.eval_integrals(TBF,TBF)}")

print("Approaching crossing")
ndim = 2
h=1.0e-2
nstep = 20000
model = BerryModel2DParallelTransport(0.02,3.0,15.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([-0.5, 0.0])
init_pvec = np.asarray([20.0, 0.0])
init_state = state
bundle.add_trajectory(init_rvec,init_pvec,init_state)
TBF = bundle.trajectorylist[0]
G = np.diag(np.reciprocal(TBF.get_masses()))
m = TBF.get_masses()

bundle.BuildHeff()
print(f"Quantum energy: {bundle.H[0,0]}")
print(f"Closer to exact Quantum energy: {model.eval_integrals(TBF,TBF)}")

print("Approaching crossing")
ndim = 2
h=1.0e-2
nstep = 20000
model = BerryModel2DParallelTransport(0.02,3.0,15.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([-0.5, 0.0])
init_pvec = np.asarray([20.0, 0.0])
init_state = state
bundle.add_trajectory(init_rvec,init_pvec,init_state)
TBF = bundle.trajectorylist[0]
G = np.diag(np.reciprocal(TBF.get_masses()))
m = TBF.get_masses()

bundle.BuildHeff()
print(f"Quantum energy: {bundle.H[0,0]}")
print(f"Closer to exact Quantum energy: {model.eval_integrals(TBF,TBF)}")

print("Near crossing")
ndim = 2
h=1.0e-2
nstep = 20000
model = BerryModel2DParallelTransport(0.02,3.0,15.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([-0.2, 0.0])
init_pvec = np.asarray([20.0, 0.0])
init_state = state
bundle.add_trajectory(init_rvec,init_pvec,init_state)
TBF = bundle.trajectorylist[0]
G = np.diag(np.reciprocal(TBF.get_masses()))
m = TBF.get_masses()

bundle.BuildHeff()
print(f"Quantum energy: {bundle.H[0,0]}")
print(f"Closer to exact Quantum energy: {model.eval_integrals(TBF,TBF)}")

print("At crossing")
ndim = 2
h=1.0e-2
nstep = 20000
model = BerryModel2DParallelTransport(0.02,3.0,15.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([0.0, 0.0])
init_pvec = np.asarray([20.0, 5.0])
init_state = state
bundle.add_trajectory(init_rvec,init_pvec,init_state)
TBF = bundle.trajectorylist[0]
G = np.diag(np.reciprocal(TBF.get_masses()))
m = TBF.get_masses()

bundle.BuildHeff()
print(f"Quantum energy: {bundle.H[0,0]}")
print(f"Closer to exact Quantum energy: {model.eval_integrals(TBF,TBF)}")
