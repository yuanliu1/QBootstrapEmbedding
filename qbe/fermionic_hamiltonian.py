"""
Source: bootstrap_05112022, bootstrap_06012022,
Reads data files to fetch the fermionic H

Chemistry team -- fermionic H for each fragment (Schmidt decomposition from total H)
"""
# Imports
import sys
import time, itertools, h5py
import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
from itertools import product
from pyscf import ao2mo

# Below has been removed from qiskit!
# from qiskit.chemistry import FermionicOperator, QMolecule
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
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

numpy_solver = NumPyMinimumEigensolver()


def fermionic_1rdm(fulldm, nao):
    """
    Inputs:
        fulldm: Full density matrix of fragment
        nao: number of atomic orbitals

    Ouput: Fermionic 1RDM
    """
    # construct the fermionic 1-rdm, Tr[a_i^\dagger a_j \rho]
    # make a qubit operator for the fermionic 1rdm to measure later
    tempferm1rdm_up = np.zeros((nao, nao))
    tempferm1rdm_dn = np.zeros((nao, nao))
    for qi in range(nao):
        for qj in range(qi, nao):
            # spin up first
            label = "+_" + str(qi) + " -_" + str(qj)
            f1rdm_labels_up = FermionicOp((label, 1.0), register_length=2 * nao, display_format="sparse")
            # transform this to qubit representation
            qubit_converter = QubitConverter(mapper=JordanWignerMapper())
            ferm1RDM_qubitop_up = qubit_converter.convert(f1rdm_labels_up, nao)
            tempferm1rdm_up[qi, qj] = np.trace(np.matmul(ferm1RDM_qubitop_up.to_matrix(), fulldm))
            tempferm1rdm_up[qj, qi] = tempferm1rdm_up[qi, qj]

            # then spin down
            label = "+_" + str(qi + nao) + " -_" + str(qj + nao)
            f1rdm_labels_dn = FermionicOp((label, 1.0), register_length=2 * nao, display_format="sparse")
            # transform this to qubit representation
            qubit_converter = QubitConverter(mapper=JordanWignerMapper())
            ferm1RDM_qubitop_dn = qubit_converter.convert(f1rdm_labels_dn)
            tempferm1rdm_dn[qi, qj] = np.trace(np.matmul(ferm1RDM_qubitop_dn.to_matrix(), fulldm))
            tempferm1rdm_dn[qj, qi] = tempferm1rdm_dn[qi, qj]

    tempferm1rdm = np.stack((tempferm1rdm_up, tempferm1rdm_dn))

    return tempferm1rdm


def fermionic_2rdm(fulldm, nao):
    """

    :param fulldm:
    :param nao:
    :return:
    """
    # construct the fermionic 2-RDMs, \Gamma_{ij,kl} = < a_i^\dagger a_k^\dagger a_l a_j >, do this for up-up spin, down-down spin, and up-down spin sector separately
    tempferm2rdm_upup = np.zeros((nao, nao, nao, nao))
    tempferm2rdm_dndn = np.zeros((nao, nao, nao, nao))
    tempferm2rdm_updn = np.zeros((nao, nao, nao, nao))
    for qi in range(nao):
        for qj in range(qi, nao):  # because permuting qi and qj does not change value
            for qk in range(nao):
                for ql in range(qk, nao):  # because permuting qk and ql does not change value
                    # spin up up first
                    label = "+_" + str(qi) + " +_" + str(qk) + " -_" + str(ql) + " -_" + str(qj)
                    f2rdm_labels_upup = FermionicOp((label, 1.0), register_length=2 * nao, display_format="sparse")
                    # transform this to qubit representation
                    qubit_converter = QubitConverter(mapper=JordanWignerMapper())
                    ferm2RDM_qubitop_upup = qubit_converter.convert(f2rdm_labels_upup, nao)
                    tempferm2rdm_upup[qi, qj, qk, ql] = np.trace(
                        np.matmul(ferm2RDM_qubitop_upup.to_matrix(), fulldm))
                    tempferm2rdm_upup[qj, qi, qk, ql] = tempferm2rdm_upup[qi, qj, qk, ql]
                    tempferm2rdm_upup[qi, qj, ql, qk] = tempferm2rdm_upup[qi, qj, qk, ql]
                    tempferm2rdm_upup[qj, qi, ql, qk] = tempferm2rdm_upup[qi, qj, qk, ql]

                    # spin down down
                    label = "+_" + str(nao + qi) + " +_" + str(nao + qk) + " -_" + str(nao + ql) + " -_" + str(nao + qj)
                    f2rdm_labels_dndn = FermionicOp((label, 1.0), register_length=2 * nao, display_format="sparse")
                    # transform this to qubit representation
                    qubit_converter = QubitConverter(mapper=JordanWignerMapper())
                    ferm2RDM_qubitop_dndn = qubit_converter.convert(f2rdm_labels_dndn, nao)
                    tempferm2rdm_dndn[qi, qj, qk, ql] = np.trace(
                        np.matmul(ferm2RDM_qubitop_dndn.to_matrix(), fulldm))
                    tempferm2rdm_dndn[qj, qi, qk, ql] = tempferm2rdm_dndn[qi, qj, qk, ql]
                    tempferm2rdm_dndn[qi, qj, ql, qk] = tempferm2rdm_dndn[qi, qj, qk, ql]
                    tempferm2rdm_dndn[qj, qi, ql, qk] = tempferm2rdm_dndn[qi, qj, qk, ql]

                    # spin up-down
                    label = "+_" + str(qi) + " +_" + str(nao + qk) + " -_" + str(nao + ql) + " -_" + str(qj)
                    f2rdm_labels_updn = FermionicOp((label, 1.0), register_length=2 * nao, display_format="sparse")
                    # transform this to qubit representation
                    qubit_converter = QubitConverter(mapper=JordanWignerMapper())
                    ferm2RDM_qubitop_updn = qubit_converter.convert(f2rdm_labels_updn, nao)
                    tempferm2rdm_updn[qi, qj, qk, ql] = np.trace(
                        np.matmul(ferm2RDM_qubitop_updn.to_matrix(), fulldm))
                    tempferm2rdm_updn[qj, qi, qk, ql] = tempferm2rdm_updn[qi, qj, qk, ql]
                    tempferm2rdm_updn[qi, qj, ql, qk] = tempferm2rdm_updn[qi, qj, qk, ql]
                    tempferm2rdm_updn[qj, qi, ql, qk] = tempferm2rdm_updn[qi, qj, qk, ql]

    tempferm2rdm = np.stack((tempferm2rdm_upup, tempferm2rdm_dndn, tempferm2rdm_updn))

    return tempferm2rdm


def frag2f1rdm(ham_qubit_mat, frag, nao):
    # given fragment number, output the fermionic 1rdm of it
    #eigval, eigvec = LA.eig(ham_qubit_mat[frag])
    eigval, eigvec = eigs(ham_qubit_mat[frag], k=1, which='SR')
    # sort the eigenvalues and the corresponding eigenvectors in descending order, last element is the ground state
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]

    #print(eigval)
    #print(eigvec[:,-1])
    print('E(ED in qubit basis, frag %d) = %.8f' % (frag, eigval[-1].real))
    # construct the total density matrix from state vector
    fulldm = DensityMatrix(eigvec[:,-1])
    ferm1rdm = fermionic_1rdm(fulldm, nao)

    return ferm1rdm
