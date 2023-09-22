# 07/19/2022, A pyscf solver for the H8 molecules in the localized basis.

import h5py,numpy
from pyscf import gto, scf, ao2mo, cc, fci, lo
from functools import reduce
from scipy.linalg import fractional_matrix_power

#from qiskit.chemistry.drivers import PySCFDriver, UnitsType
#from qiskit.chemistry import FermionicOperator, QMolecule
from qiskit.algorithms.phase_estimators.hamiltonian_phase_estimation import HamiltonianPhaseEstimation
from qiskit.opflow import SummedOp
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import QuantumInstance
import numpy,time,itertools,h5py
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



def get_lo_ints(mol):
    ''' Given mol object, outputs the 1- and 2-e integrals in the LO basis'''

    C_lo = lo.orth_ao(mol, 'lowdin')
    norb = 8
    h1_lo = numpy.matmul(C_lo.T, numpy.matmul(scf.hf.get_hcore(mol), C_lo))
    eri_lo = ao2mo.kernel(mol, C_lo, aosym = 1)
    eri_lo = numpy.reshape(eri_lo, (norb,norb,norb,norb))

    return h1_lo, eri_lo

def make_mean_field(h1, eri, nele_up, nele_dn = None):
    ''' make a fake object mole from given h1 and eri, in orthogonal basis
        Input: mol_name: string with the molecule name
        Output: mf: mean field object from pyscf
    '''

    assert h1.shape[0] == h1.shape[1] and h1.shape[0] == eri.shape[0]
    nao = h1.shape[0]

    if nele_dn == None:    
        nele = 2* nele_up
    else:
        nele = nele_up + nele_dn

    mol = gto.M()
    mol.nelectron = nele

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: numpy.eye(nao)
    mf._eri = eri

    return mf



################# Main Algorithm ####################
mol_name = 'H8_1A_'
mol_name_lo = 'H8_1A_lo_'
neup = 4
nedown = 4
index1e = 'i0f0'
index2e = 'f0'
file_h1 = h5py.File('data/' + mol_name + 'h1.h5','w')
file_h2 = h5py.File('data/' + mol_name + 'eri_file.h5','w')
file_h1_lo = h5py.File('data/' + mol_name_lo + 'h1.h5','w')
file_h2_lo = h5py.File('data/' + mol_name_lo + 'eri_file.h5','w')

#mol_h2 = gto.M(atom = 'H 0.0 0.0 0.0; H 0.0 0.0 0.7; H 0.0 0.0 1.4; H 0.0 0.0 2.1', basis = 'sto-3g')
#mol_h2 = gto.M(atom = 'H 0.0 0.0 0.0; H 0.0 0.0 0.8; H 0.0 0.0 1.6; H 0.0 0.0 2.4', basis = 'sto-3g')
#mol_h2 = gto.M(atom = 'H 0.0 0.0 0.0; H 0.0 0.0 0.9; H 0.0 0.0 1.8; H 0.0 0.0 2.7', basis = 'sto-3g')
#mol_h2 = gto.M(atom = 'H 0.0 0.0 0.0; H 0.0 0.0 1.0; H 0.0 0.0 2.0; H 0.0 0.0 3.0', basis = 'sto-3g')
mol_h2 = gto.M(atom = 'H 0.0 0.0 0.0; H 0.0 0.0 1.0; H 0.0 0.0 2.0; H 0.0 0.0 3.0; H 0.0 0.0 4.0; H 0.0 0.0 5.0; H 0.0 0.0 6.0; H 0.0 0.0 7.0', basis = 'sto-3g')


print("\n\n########### Standard PySCF calculations for benchmark")
mf = scf.RHF(mol_h2)
mf.kernel() 
print(mf.energy_nuc())
    
cisolver = fci.FCI(mf)
#cisolver.verbose = 4
e, fcivec = cisolver.kernel()
#print(cisolver.kernel())
print('E(FCI) = %.6f' % e)
print('E(FCI, no nuclei) = %.6f' % (e - mf.energy_nuc()))


print("\n\n###### Generate the integrals in the LO basis and check the energy")
h1_lo, eri_lo = get_lo_ints(mol_h2)
#print("h1_lo = ", h1_lo)
#print("eri_lo = ", eri_lo)
mf_lo = make_mean_field(h1_lo, eri_lo, neup, nedown)
mf_lo.kernel() 
cisolver_lo = fci.FCI(mf_lo)
#cisolver.verbose = 4
e_lo, fcivec_lo = cisolver_lo.kernel()
#print(cisolver.kernel())
print('E(FCI) = %.6f' % e_lo)

file_h1_lo.create_dataset(index1e, data = h1_lo)
file_h2_lo.create_dataset(index2e, data = eri_lo)
file_h1_lo.close()
file_h2_lo.close()


#print("\n\n###### Map to qubit basis and check the energy")
#electronic_energy_from_ints = ElectronicEnergy.from_raw_integrals(
#    ElectronicBasis.MO, h1_lo, eri_lo
#)
#
#ferOp = electronic_energy_from_ints.second_q_ops()[0]  
#qubit_converter = QubitConverter(mapper = JordanWignerMapper())
#qubitOp = qubit_converter.convert(ferOp) # this will not give two-qubit reduction
##print(qubitOp)
#print('number of total qubits: %d' % qubitOp.num_qubits)
#ham = qubitOp.to_matrix()
##print(ham)
#
#eigval, eigvec = LA.eig(ham)
##print(min(eigval))
## sort the eigenvalues and the corresponding eigenvectors in descending order, last element is the ground state
#idx = eigval.argsort()[::-1]   
#eigval = eigval[idx]
#eigvec = eigvec[:,idx]
#
##print(eigval)
##print(eigvec[:,-1])
#print('E (ED in qubit basis) = %.8f' % eigval[-1].real)





#print("\n\n####### Form the unitary block encoding matrix")
#alpha = 2.0
#ham = ham/alpha
#pham = fractional_matrix_power(numpy.eye(16) - numpy.matmul(ham, ham), 0.5)
##print(pham)
#
#ube0 = numpy.concatenate((ham, pham), axis=0)
##print("shape of ube0", ube0.shape)
#assert numpy.allclose(ube0[:16,:16], ham)
#assert numpy.allclose(ube0[16:,:16], pham)
#
#ube1 = numpy.concatenate((pham, -ham), axis=0)
##print("shape of ube1", ube1.shape)
#ube = numpy.matrix(numpy.concatenate((ube0, ube1), axis=1))
##print("shape of ube", ube.shape)
##print(ube)
#assert numpy.allclose(numpy.matmul(ube, ube.H), numpy.eye(32))
#
#numpy.save('ube_h4_alpha2.npy', ube)
print("OK!")
