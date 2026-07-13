import copy
import sys
import math
import os
import re
import numpy as np
import cmath
import random

from gaussfuncs import *

# Returns the overlap matrix S for a given Gaussian basis set
def BuildOverlapMat(basis_set):
  N_TBF = len(basis_set)
  S = np.zeros((N_TBF,N_TBF),dtype=complex)
  for idx1,basis_func_1 in enumerate(basis_set):
    for idx2,basis_func_2 in enumerate(basis_set):
      S[idx1,idx2] = OverlapR(basis_func_1,basis_func_2,False)
  return S

def BuildMixedOverlapMat(basis_set_1,basis_set_2):
  N_TBF_1 = len(basis_set_1)
  N_TBF_2 = len(basis_set_2)
  S = np.zeros((N_TBF_1,N_TBF_2),dtype=complex)
  for idx1,basis_func_1 in enumerate(basis_set_1):
    for idx2,basis_func_2 in enumerate(basis_set_2):
      S[idx1,idx2] = OverlapR(basis_func_1,basis_func_2,False)
  return S

# Returns the regularizer matrix: the inverse of the overlap matrix 
# with the eigenvectors associated with singularities projected out
def BuildRegInvMat(S,thresh=1.0e-4):
  N_TBF = S.shape[0]
  w,v = np.linalg.eigh(S)
 
  # sort eigenvalues and eigenvectors 
  idx = w.argsort()[::-1]
  w = w[idx]
  v = v[:,idx]

  w_abs = w.real # assuming Hermitian overlap matrix, all eigenvalues are real
  w_truncated = w_abs[w_abs > thresh]
  N_indep = len(w_truncated)
  N_lindep = N_TBF - N_indep
  I_trunc = np.eye(N_TBF)
  I_trunc[N_indep:,N_indep:] = 0.0
  return v @ I_trunc @ np.diag(np.reciprocal(w_abs)) @ I_trunc @ v.T

# Returns the inverse sqrt matrix S^(-0.5) for a given overlap matrix
# such that S^(-0.5) @ S @ S^(-0.5)T = I
def BuildSinvp5Mat(S,thresh=1.0e-4):
  N_TBF = S.shape[0]
  w,v = np.linalg.eigh(S)
 
  # sort eigenvalues and eigenvectors 
  idx = w.argsort()[::-1]
  w = w[idx]
  v = v[:,idx]

  w_abs = w.real # assuming Hermitian overlap matrix, all eigenvalues are real
  w_truncated = w_abs[w_abs > thresh]
  N_indep = len(w_truncated)
  N_lindep = N_TBF - N_indep
  I = np.eye(N_TBF)
  I_trunc = I[:N_indep,:]
  return np.diag(np.sqrt(np.reciprocal(w_truncated))) @ I_trunc @ v.T

def BuildHamiltonianMat(basis_set,V,b_SPA0,args_dict):
  N_TBF = len(basis_set)
  H = np.zeros((N_TBF,N_TBF),dtype=complex)
  for idx1,basis_func_1 in enumerate(basis_set):
    for idx2,basis_func_2 in enumerate(basis_set):
      if (idx2 < idx1):
        H[idx1,idx2] = H[idx2,idx1].conjugate() 
      else:
        H[idx1,idx2] = Hamiltonian(V,basis_func_1,basis_func_2,False,b_SPA0,args_dict)
  return H

def PrintCompMat(A):
  N1 = A.shape[0]
  N2 = A.shape[1]
  for i in range(N1):
    for j in range(N2):
      print("({:16.10f},{:16.10f})".format(A[i,j].real,A[i,j].imag),end="")
    print("")

def PrintCompVec(A):
  N = A.shape[0]
  for i in range(N):
    print("({:16.10f},{:16.10f})".format(A[i].real,A[i].imag),end="")
  print("")

def SaveCompMat(A,filename):
  N1 = A.shape[0]
  N2 = A.shape[1]
  with open("{}_real.dat".format(filename),'w+') as f:
    for i in range(N1):
      for j in range(N2):
        print("{:16.10f}".format(A[i,j].real),end="",file=f)
      print("",file=f)
  with open("{}_imag.dat".format(filename),'w+') as f:
    for i in range(N1):
      for j in range(N2):
        print("{:16.10f}".format(A[i,j].imag),end="",file=f)
      print("",file=f)

def LoadCompMat(filename):
  return np.loadtxt("{}_real.dat".format(filename)) + 1j*np.loadtxt("{}_imag.dat".format(filename))
