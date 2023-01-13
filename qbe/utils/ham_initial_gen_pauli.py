# 10/26/2022, this script read the integrals from files, map to qubit Pauli basis, and then print the results


from qiskit.algorithms.phase_estimators.hamiltonian_phase_estimation import HamiltonianPhaseEstimation
from qiskit.opflow import SummedOp
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import QuantumInstance
import numpy, time, itertools, h5py
from numpy import linalg as LA
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

numpy_solver = NumPyMinimumEigensolver()

# only use the following two lines on slurm
import matplotlib

matplotlib.use('pdf')

import matplotlib.pylab as plt
import sys
# from subsys_ovlp.py import SubsysOvlp
from datetime import datetime

start_time = datetime.now()

# Whether to read in MO or AO integrals. 0 for AO, 1 for MO. For local information used in the matching, we should set read_ints_basis = 0 to use AO basis.
# Here "AO" is defined as the initial basis where bootstrap embedding Hamiltonians are obtained. While "MO" is defined as the new orbitals defined by the HF solutions of the BE hamiltonians.
read_ints_basis = 0

# whether to diagonalize to check eigenvalues
flag_check_eig = False

# mol_name = 'H2_1A_mo_'
# num_frag = 1 # number of fragments
# nao=2  # nao - no. of orbs

# mol_name = 'H4_0d9A_lo_'
# num_frag = 1 # number of fragments
# nao=4  # nao - no. of orbs

# mol_name = 'H4_'
# mol_name = 'H4_pert_'
# num_frag = 2 # number of fragments
# nao=4  # nao - no. of orbs


# mol_name = 'H4_1A_lo_'
# num_frag = 1 # number of fragments
# nao=4  # nao - no. of orbs

## This is the global H8 Hamiltonian, not the fragment Hamiltonians
# mol_name = 'H8_1A_lo_'
# num_frag = 1 # number of fragments
# nao=8  # nao - no. of orbs


# mol_name = 'H6_'
# num_frag = 4 # number of fragments
# nao = 6  # nao - no. of orbs

## This is the fragment Hamiltonian
mol_name = 'H8_'
# mol_name = 'H8_pert_'
num_frag = 6  # number of fragments
nao = 6  # nao - no. of orbs

num_it = 1

with open('data/' + mol_name + 'initial_ham_pauli.txt', 'w') as file_ham_pauli:
    for it in range(num_it):
        for frag in range(num_frag):
            index1e = 'i' + str(it) + 'f' + str(frag)
            index2e = 'f' + str(frag)
            print('***' + index1e + '***')
            file_ham_pauli.write('***' + index1e + '***\n')

            # access 1e- integral
            if read_ints_basis == 0:
                r = h5py.File('data/' + mol_name + 'h1.h5', 'r')
            else:
                r = h5py.File('data/' + mol_name + 'h1_mo.h5', 'r')
            h1 = numpy.array(r.get(index1e))
            # print(h1)
            r.close()

            # access 2e- integral
            if read_ints_basis == 0:
                r = h5py.File('data/' + mol_name + 'eri_file.h5', 'r')
            else:
                r = h5py.File('data/' + mol_name + 'eri_mo_file.h5', 'r')

            # r.keys()
            h2 = numpy.array(r.get(index2e))
            r.close()
            # print(h2)

            ## Change from 4-fold to no symmetry
            symm = 1  # change to 1 for no symmetry
            # print(h2)
            eri = ao2mo.restore(symm, h2, nao)

            ## Try to get rid of last spatial orbital
            # nao=5
            # h1 = h1[:-1, :-1]
            # eri = eri[:-1, :-1, :-1, :-1]

            electronic_energy_from_ints = ElectronicEnergy.from_raw_integrals(
                ElectronicBasis.MO, h1, eri.reshape((nao, nao, nao, nao))
            )
            # print(electronic_energy_from_ints)

            ferOp = electronic_energy_from_ints.second_q_ops()[0]  # here, output length is always 1
            # es_problem = ElectronicStructureProblem(electronic_energy_from_ints)
            qubit_converter = QubitConverter(mapper=JordanWignerMapper())
            # qubit_converter = QubitConverter(mapper = ParityMapper(), two_qubit_reduction = True)
            qubitOp = qubit_converter.convert(ferOp)  # this will not give two-qubit reduction
            # qubitOp = qubit_converter.convert(ferOp, nao) # use this and the above to get two-qubit reduction
            print(qubitOp[0])
            print('number of total qubits: %d \n' % qubitOp.num_qubits)

            file_ham_pauli.write("%s\n\n" % qubitOp)

    print('Done')



