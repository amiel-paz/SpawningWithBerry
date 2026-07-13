import os, math
import numpy as np
import copy
import onedgauss

# A multi-dimensional Gaussian TBF (uncoupled dimensions)
class Trajectory(object):

    def __init__(self, ndim):
        self.ndim = ndim # dimensionality of N-dim TBF 
        self.rvec = np.zeros(ndim) # List of all Gaussian centroid positions
        self.pvec = np.zeros(ndim) # List of all Gaussian centroid momenta
        self.widths = np.ones(ndim) # List of all Gaussian widths
        self.masses = np.ones(ndim) * 1000 # List of all masses
        self.state = None # Current electronic state index
        
        self.gausslist = [] # List of all Gaussians
        for idx in range(self.ndim):
            self.gausslist.append(onedgauss.OneD_Gauss(0.0,0.0,1.0,1000,0.0,1.0)) 

    def get_widths(self):
        return self.widths    

    def get_masses(self):
        return self.masses 

    def get_rvec(self):
        return self.rvec
 
    def get_pvec(self): 
        return self.pvec

    def get_state(self):
        return self.state

    def set_widths(self, widths):
        assert len(widths) == self.ndim, "Must be a vector of length self.ndim" 
        self.widths = copy.deepcopy(widths)
        for idx in range(self.ndim):
            self.gausslist[idx].update_a(widths[idx])

    def set_masses(self, masses):
        assert len(masses) == self.ndim, "Must be a vector of length self.ndim" 
        self.masses = copy.deepcopy(masses)
        for idx in range(self.ndim):
            self.gausslist[idx].update_m(masses[idx])

    def set_rvec(self, rvec):
        assert len(rvec) == self.ndim, "Must be a vector of length self.ndim" 
        self.rvec = copy.deepcopy(rvec)
        for idx in range(self.ndim):
            self.gausslist[idx].update_R(rvec[idx])
     
    def set_pvec(self, pvec): 
        assert len(pvec) == self.ndim, "Must be a vector of length self.ndim" 
        self.pvec = copy.deepcopy(pvec)
        for idx in range(self.ndim):
            self.gausslist[idx].update_P(pvec[idx])

    def set_state(self, state):
        self.state = state
