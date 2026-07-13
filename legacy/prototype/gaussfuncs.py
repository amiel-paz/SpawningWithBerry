import copy
import sys
import math
import os
import re
import numpy as np
import cmath
import random

from complexint import *
from onedgauss import *

# This function computes the equivalent of <chi1|chi2> or, int_-\infty^\infty chi1^* chi2 dR
def OverlapR(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Begin calculation
    a = a1 + a2
    Rcent = (a1*R1 + a2*R2)/a
    acent = (a1*a2)/a
    prefactor = (2/a)**0.5 * (a1*a2)**0.25
    if (b_Coeffs):
      prefactor *= coeff1.conjugate() * coeff2
    B = (P2*R2 - P1*R1) - (P2 - P1)*Rcent
    A = acent * (R2-R1)**2 + (1/(4*a)) * (P2-P1)**2
    C = gamma2 - gamma1
    return prefactor * cmath.exp(-1j*B) * math.exp(-A) * cmath.exp(1j*C)

# This function computes the equivalent of <chi1|chi2> or, int_-\infty^\infty chi1^* chi2 dP
def OverlapP(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Begin calculation
    # Transform invwidths to momentum space
    a1 = 1/(4*a1)
    a2 = 1/(4*a2)
    a = a1 + a2
    Pcent = (a1*P1 + a2*P2)/a
    acent = (a1*a2)/a
    prefactor = (2/a)**0.5 * (a1*a2)**0.25
    if (b_Coeffs):
      prefactor *= coeff1.conjugate() * coeff2
    B = - (R1 - R2)*Pcent
    A = acent * (P2-P1)**2 + (1/(4*a)) * (R1-R2)**2
    C = gamma2 - gamma1
    return prefactor * cmath.exp(-1j*B) * math.exp(-A) * cmath.exp(1j*C)

def NumOverlap(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Populate intermediate quantities
    a = a1 + a2
    Rcent = (a1*R1 + a2*R2)/a
    
    # Compute main expression
    thresh = 1.0e-20
    lb = Rcent - math.sqrt(-math.log(thresh)/a)
    ub = Rcent + math.sqrt(-math.log(thresh)/a)
    return complex_quadrature(lambda x: chi1.get_values_conjugate(x)*chi2.get_values(x), lb, ub)[0]

# This function computes the equivalent of <chi1|d/dx|chi2>
def Overlap_dR(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Populate intermediate quantities
    delta_R = R2 - R1
    P_12 = a1*P1 + a2*P2

    # Begin calculation
    dR_S_12 = (P_12*1j - 2.0*a2*a1*delta_R)/(a1 + a2)
    return dR_S_12 * OverlapR(chi1,chi2,b_Coeffs) 

# This function computes the equivalent of <chi1|d/dp|chi2>
def Overlap_dP(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Populate intermediate quantities
    delta_R = R2 - R1
    delta_P = P2 - P1

    # Begin calculation
    dP_S_12 = (delta_P + 2.0j * a2 * delta_R)/(2.0 * (a1 + a2))
    return dP_S_12 * OverlapR(chi1,chi2,b_Coeffs)

# This function computes the equivalent of <chi1|d/dt|chi2>
def OverlapDot(dVdx, chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Populate intermediate quantities
    v2 = P2/chi2.m
    F2 = dVdx(R2)

    # Begin calculation
    return -v2*Overlap_dR(chi1,chi2,b_Coeffs) + F2*Overlap_dP(chi1,chi2,b_Coeffs)

# This function computes the equivalent of <chi1|d/dt|chi2>,
# assuming dR/dt, dP/dt, da/dt, dgamma/dt all have specific forms
def OverlapDotGeneral(chi1, chi2, dR_dt, dP_dt, da_dt, dgamma_dt, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Begin calculation. Have not worked out thawed Gaussian quantities yet.
    return dR_dt * -Overlap_dR(chi1,chi2,b_Coeffs) + dP_dt * Overlap_dP(chi1,chi2,b_Coeffs)

# This function computes the equivalent of <chi1|H|chi2>
def Hamiltonian(V, chi1, chi2, b_Coeffs, b_SPA0, args_dict):

    KE = Kinetic(chi1, chi2, b_Coeffs)
    PE = Potential(V, chi1, chi2, b_Coeffs, b_SPA0, args_dict)
    return KE + PE

# This function computes the equivalent of <chi1|T|chi2>
def Kinetic(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    m1 = chi1.m
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    m2 = chi2.m
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    mred = (m1 + m2)/2

    # Begin calculation
    # Transform invwidths to momentum space
    a1 = 1/(4*a1)
    a2 = 1/(4*a2)
    a = a1 + a2
    Pcent = (a1*P1 + a2*P2)/a
    acent = (a1*a2)/a
    prefactor = (2/a)**0.5 * (a1*a2)**0.25
    if (b_Coeffs):
      prefactor *= coeff1.conjugate() * coeff2
    prefactor *= ( (1/(2*a) - ((R1-R2)**2 / (4*a*a))) + 2j * Pcent * ((R1-R2)/(2*a)) + Pcent**2) / (2*mred)
    B = - (R1 - R2)*Pcent
    A = acent * (P2-P1)**2 + (1/(4*a)) * (R1-R2)**2
    C = gamma2 - gamma1
    return prefactor * cmath.exp(-1j*B) * math.exp(-A) * cmath.exp(1j*C)

def NumKinetic1(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    m1 = chi1.m
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    m2 = chi2.m
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Populate intermediate quantities
    a = a1 + a2
    Rcent = (a1*R1 + a2*R2)/a
    mred = (m1 + m2)/2
    
    # Compute main expression
    thresh = 1.0e-20
    lb = Rcent - math.sqrt(-math.log(thresh)/a)
    ub = Rcent + math.sqrt(-math.log(thresh)/a)
    return complex_quadrature(lambda x: chi1.get_values_conjugate(x)*chi2.get_values(x)*(-2*a1*(x-R1) - 1j*P1)*(-2*a2*(x-R2) + 1j*P2), lb, ub)[0] / (2*mred)

def NumKinetic2(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    m1 = chi1.m
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    m2 = chi2.m
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Populate intermediate quantities
    a = a1 + a2
    Rcent = (a1*R1 + a2*R2)/a
    mred = (m1 + m2)/2
    
    # Compute main expression
    thresh = 1.0e-20
    lb = Rcent - math.sqrt(-math.log(thresh)/a)
    ub = Rcent + math.sqrt(-math.log(thresh)/a)
    return complex_quadrature(lambda x: -chi1.get_values_conjugate(x)*chi2.get_values(x)*(-2*a2 + (-2*a2*(x-R2) + 1j*P2)*(-2*a2*(x-R2) + 1j*P2)), lb, ub)[0] / (2*mred)

def NumKinetic3(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    m1 = chi1.m
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    m2 = chi2.m
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Populate intermediate quantities
    a = a1 + a2
    a_tilde_1 = 1/(4*a1)
    a_tilde_2 = 1/(4*a2)
    a_tilde = a_tilde_1 + a_tilde_2
    Pcent = (a_tilde_1*P1 + a_tilde_2*P2)/a_tilde
    mred = (m1 + m2)/2
    
    # Compute main expression
    thresh = 1.0e-20
    lb = Pcent - math.sqrt(-math.log(thresh)/a_tilde)
    ub = Pcent + math.sqrt(-math.log(thresh)/a_tilde)
    return complex_quadrature(lambda p: chi1.get_ft_values_conjugate(p)*chi2.get_ft_values(p)*p*p, lb, ub)[0] / (2*mred)

# This function computes the equivalent of <chi1|p|chi2>
def Momentum(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Begin calculation
    # Transform invwidths to momentum space
    a1 = 1/(4*a1)
    a2 = 1/(4*a2)
    a = a1 + a2
    Pcent = (a1*P1 + a2*P2)/a
    acent = (a1*a2)/a
    prefactor = (2/a)**0.5 * (a1*a2)**0.25
    if (b_Coeffs):
      prefactor *= coeff1.conjugate() * coeff2
    prefactor *=  (1j*(R1-R2)/(2*a) + Pcent)
    B = - (R1 - R2)*Pcent
    A = acent * (P2-P1)**2 + (1/(4*a)) * (R1-R2)**2
    C = gamma2 - gamma1
    return prefactor * cmath.exp(-1j*B) * math.exp(-A) * cmath.exp(1j*C)

def NumMomentum1(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    m1 = chi1.m
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    m2 = chi2.m
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Populate intermediate quantities
    a = a1 + a2
    Rcent = (a1*R1 + a2*R2)/a
    mred = (m1 + m2)/2
    
    # Compute main expression
    thresh = 1.0e-20
    lb = Rcent - math.sqrt(-math.log(thresh)/a)
    ub = Rcent + math.sqrt(-math.log(thresh)/a)
    return complex_quadrature(lambda x: -1j*chi1.get_values_conjugate(x)*chi2.get_values(x)*(-2*a2*(x-R2) + 1j*P2), lb, ub)[0]

def NumMomentum2(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    m1 = chi1.m
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    m2 = chi2.m
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Populate intermediate quantities
    a = a1 + a2
    Rcent = (a1*R1 + a2*R2)/a
    mred = (m1 + m2)/2
    
    # Compute main expression
    thresh = 1.0e-20
    lb = Rcent - math.sqrt(-math.log(thresh)/a)
    ub = Rcent + math.sqrt(-math.log(thresh)/a)
    return complex_quadrature(lambda x: 1j*chi1.get_values_conjugate(x)*chi2.get_values(x)*(-2*a1*(x-R1) - 1j*P1), lb, ub)[0]

def NumMomentum3(chi1, chi2, b_Coeffs):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    m1 = chi1.m
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    m2 = chi2.m
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Populate intermediate quantities
    a = a1 + a2
    a_tilde_1 = 1/(4*a1)
    a_tilde_2 = 1/(4*a2)
    a_tilde = a_tilde_1 + a_tilde_2
    Pcent = (a_tilde_1*P1 + a_tilde_2*P2)/a_tilde
    mred = (m1 + m2)/2
    
    # Compute main expression
    thresh = 1.0e-20
    lb = Pcent - math.sqrt(-math.log(thresh)/a_tilde)
    ub = Pcent + math.sqrt(-math.log(thresh)/a_tilde)
    return complex_quadrature(lambda p: chi1.get_ft_values_conjugate(p)*chi2.get_ft_values(p)*p, lb, ub)[0]

# This function computes the equivalent of <chi1|V|chi2>
def Potential(V, chi1, chi2, b_Coeffs, b_SPA0, args_dict):

    # Populate variables
    R1 = chi1.R
    P1 = chi1.P
    a1 = chi1.a
    gamma1 = chi1.gamma
    coeff1 = chi1.coeff
    R2 = chi2.R
    P2 = chi2.P
    a2 = chi2.a
    gamma2 = chi2.gamma
    coeff2 = chi2.coeff

    # Populate intermediate quantities
    a = a1 + a2
    Rcent = (a1*R1 + a2*R2)/a

    if (b_SPA0):
      return V(Rcent,args_dict) * Overlap(chi1,chi2,b_Coeffs)
    else:
      thresh = 1.0e-20
      lb = Rcent - math.sqrt(-math.log(thresh)/a)
      ub = Rcent + math.sqrt(-math.log(thresh)/a)
      return complex_quadrature(lambda x: chi1.get_values_conjugate(x)*V(x,args_dict)*chi2.get_values(x), lb, ub)[0]

def GenGaussFit(V, args_dict, lower_bound, upper_bound, N_points=1000, thresh=1.0e-6):
  gauss_centroids = np.linspace(lower_bound, upper_bound, N_points)
  gaussian_basis_set = []  
  for i_gaussian in range(N_points):
    Psi = OneD_Gauss(gauss_centroids[i_gaussian],      # position
          0.0,                      # momentum
         N_points/(upper_bound - lower_bound),   # invwidth
          1.0,                      # mass
          0.0,                      # QM phase
          1.0)                      # coeff
    gaussian_basis_set.append(Psi)
  K = np.zeros((len(gaussian_basis_set),len(gauss_centroids)),dtype=complex)
  for i_gaussian,gaussian in enumerate(gaussian_basis_set):
    for i_centroid,centroid in enumerate(gauss_centroids):
      K[i_gaussian,i_centroid] = gaussian.get_values(centroid)
  weights = np.linalg.pinv(K,thresh) @ V(gauss_centroids, args_dict)
  print("Residual of fit: {:16.10f}".format(np.linalg.norm(V(gauss_centroids,args_dict) - K @ weights)))
  return weights, gaussian_basis_set

def MaxOverlapGamma(chi1, chi2):
  return -(1j/(4*(chi1.a + chi2.a)))*(chi1.P - chi2.P)**2 - 1j*cmath.log(chi2.coeff.conjugate())

def MaxOverlapGammaNoCoeff(chi1, chi2):
  return -(1j/(4*(chi1.a + chi2.a)))*(chi1.P - chi2.P)**2 
