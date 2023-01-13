# imports
from qiskit.algorithms.phase_estimators.hamiltonian_phase_estimation import HamiltonianPhaseEstimation
from qiskit.opflow import SummedOp
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import QuantumInstance
import numpy,time,itertools,h5py
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
from itertools import product
from pyscf import ao2mo
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import partial_trace, DensityMatrix

from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicEnergy,
    ElectronicDipoleMoment,
    ParticleNumber,
    AngularMomentum,
    Magnetization,
)
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
    IntegralProperty,
)
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.results import ElectronicStructureResult
from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from scipy.optimize import minimize

numpy_solver = NumPyMinimumEigensolver()


# only use the following two lines on slurm
import matplotlib
matplotlib.use('pdf')

import matplotlib.pylab as plt
import sys
#from subsys_ovlp.py import SubsysOvlp
from datetime import datetime

DATA_DIR = '../data/'


def qubit_calc_ovlp(qubitrdma, qubitrdmb, ie=2, ic=1):
  # Compute the overlap of the RDM edge sites of A (ie) with center sites of B (ic),
  # Default, assume A is to the left of B, ie = 2, ic = 1

  tempovlp = 0.0
  numsite = int(nao / 2)
  for ispin in range(2):
    tempovlp = tempovlp + numpy.add.reduce(
      numpy.matmul(qubitrdma[ispin * numsite + ie] - qubitrdmb[ispin * numsite + ic],
                   qubitrdma[ispin * numsite + ie] - qubitrdmb[ispin * numsite + ic]), axis=(0, 1))
  return tempovlp


def qubit_calc_grad(frag, qubitrdms, num_nb, nborfrag_list, icenter = 1):
  """
  :param frag:
  :param qubitrdms:
  :param num_nb:
  :param nborfrag_list:
  :param icenter: (Default=1 for a chain with 3 sites on a fragment)
  :return:
  """
  # qubit version
  ## Compute the gradient matrix for the current fragment by looping over all its adjacent fragments

  qubitrdmA = qubitrdms[frag]
  grad = numpy.zeros((num_nb[frag]))
  # Loop over all fragments that overlap with the current fragment, and compute the overlap to get the gradient
  # print(num_nb[0])
  for inb in range(num_nb[frag]):
    # ferm1rdmB = frag2f1rdm(nborfrag_list[frag][inb])
    qubitrdmB = qubitrdms[nborfrag_list[frag][inb]]

    # get the gradient
    ovlpsiteA = icenter + (nborfrag_list[frag][inb] - frag)
    grad[inb] = qubit_calc_ovlp(qubitrdmA, qubitrdmB, ovlpsiteA, icenter)

  return grad


def cost_func(mylambda):
  # compute the total RMS error as the cost function

  update_ham_f1rdm(mylambda)
  error = 0.0
  for frag in range(num_frag):
    error = error + (LA.norm(calc_grad(frag)))**2
  error = numpy.sqrt(error / n_const)
  return error


# Optimize the cost function
def min_cost_func(mylmbd):
  minimize(cost_func, mylmbd, method="BFGS", options = {'gtol': THRES_GRAD, 'eps': 1e-08, 'maxiter': None, 'disp': True, 'return_all': True, 'finite_diff_rel_step': None})
