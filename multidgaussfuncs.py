import os, math
import numpy as np
import matplotlib.pyplot as plt
from math import erf, pi, sqrt, exp
from scipy import special
from trajectory import Trajectory
from gaussfuncs import *

# Works with anything not potential-specific
def ApplyToTraj(traj1, traj2, gaussfunc, b_Coeff):
    assert traj1.ndim == traj2.ndim, "Number of dimensions in both trajectories must be the same."
    
    ndim = traj1.ndim
    value = 1.0
    for I in range(ndim):
        G_T1 = traj1.gausslist[I]
        G_T2 = traj2.gausslist[I]
        value *= gaussfunc(G_T1, G_T2, b_Coeff)
    return value

def ApplyToIndividual(traj1, traj2, gaussfunc, b_Coeff):
    assert traj1.ndim == traj2.ndim, "Number of dimensions in both trajectories must be the same."
    
    ndim = traj1.ndim
    value = []
    for I in range(ndim):
        G_T1 = traj1.gausslist[I]
        G_T2 = traj2.gausslist[I]
        value.append(gaussfunc(G_T1, G_T2, b_Coeff))
        #print(f"Dim {I+1}: {value[I]}")
    return value

def SDotTrajIndividual(traj1, traj2, dR_dt_vec, dP_dt_vec, da_dt_vec, dgamma_dt_vec, b_Coeff):
    assert traj1.ndim == traj2.ndim, "Number of dimensions in both trajectories must be the same."
    
    ndim = traj1.ndim
    value = []
    for I in range(ndim):
        G_T1 = traj1.gausslist[I]
        G_T2 = traj2.gausslist[I]
        value.append(OverlapDotGeneral(G_T1, G_T2, dR_dt_vec[I], dP_dt_vec[I], da_dt_vec[I], dgamma_dt_vec[I], b_Coeff))
        #print(f"Dim {I+1}: {value[I]}")
    return value
