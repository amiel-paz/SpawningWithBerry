from bundle import *
from models import *
from classprop import *
from testutils import *

print("At crossing")
ndim = 2
model = BerryModel2DParallelTransport(0.02,3.0,15.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([-0.05, 0.05])
init_pvec = np.asarray([20.0, -0.1])
init_state = 0
bundle.add_trajectory(init_rvec,init_pvec,init_state)
init_rvec = np.asarray([0.05, -0.05])
init_pvec = np.asarray([18.0, 0.3])
init_state = 1
bundle.add_trajectory(init_rvec,init_pvec,init_state)
TBF1 = bundle.trajectorylist[0]
TBF2 = bundle.trajectorylist[1]
G = np.diag(np.reciprocal(TBF2.get_masses()))
m = TBF2.get_masses()

bundle.BuildHeff()
print(f"Quantum energy: {bundle.H[0,1]}")
print(f"Closer to exact Quantum energy: {model.eval_integrals(TBF1,TBF2,KE=True)}")
