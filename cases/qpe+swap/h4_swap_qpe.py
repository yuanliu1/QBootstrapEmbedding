# 11/6/2022, implement 2-QBE on H_n of different bond length, followed by a swap test to estimate the overlap
# 11/7/2022, all bug fixed. results are correct now for self overlap.

from qiskit_nature.settings import settings
import qiskit.quantum_info as qi
settings.dict_aux_operators = True
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.algorithms import PhaseEstimationScale, HamiltonianPhaseEstimation
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit.utils import QuantumInstance
from qiskit.extensions import HamiltonianGate, UnitaryGate
from qiskit.compiler import transpile, assemble
from qiskit.transpiler import InstructionDurations
from qiskit.visualization import circuit_drawer, plot_histogram
from qiskit.opflow import SummedOp, PauliOp, PauliSumOp, PauliTrotterEvolution
from qiskit.test.mock import FakeMumbai
import numpy, time, itertools, h5py, math
from numpy import linalg
from itertools import product
from pyscf import ao2mo, gto, scf
from functools import reduce
from scipy.linalg import logm, expm, block_diag, eigvals, eig, eigh

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
import bisect

# only use the following two lines on slurm
import matplotlib
matplotlib.use('pdf')
import sys

import matplotlib.pylab as plt
#from subsys_ovlp.py import SubsysOvlp
from datetime import datetime


def get_mo_coeff(h1, h2, nele = 2):
    '''
    Read in the 1e- and 2e-integrals in LO and then generate 
    the orbital rotation unitary matrix to go from LOs to MOs.
    inputs: 1e- and 2e integrals, number of electrons of the molecule
    output: the orbital rotation unitary matrix from LOs to MOs
    '''

    symm = 8 # change to 8 for 8-fold symmetry
    nao = h1.shape[0]
    eri = ao2mo.restore(symm, h2, nao)
    
    # print(h2)
    
    mol = gto.M()
    # mol.nelectron = nele
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: numpy.eye(nao)
    mf._eri = eri
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    
    #mf.kernel(max_cycle = 1)
    mf.kernel()

    # check if mo_coeff is indeed a unitary
    c = mf.mo_coeff
    assert numpy.allclose(reduce(numpy.dot, (c.T, c)), numpy.eye(nao))

    return block_diag(c.T,c.T)

def get_hermitian_generator(u_rot):
    '''
    get the hermitian generator h, where mo_coeff = exp(ih)
    '''

    return logm(u_rot) / 1j


def get_pauli_sum(hgen):
    '''
    get the Pauli sum representation of the unitary transformation for quantum states 
    '''
    nao = hgen.shape[0]
    electronic_energy_from_ints = ElectronicEnergy.from_raw_integrals(
        ElectronicBasis.SO, hgen, numpy.zeros((nao, nao, nao, nao))
    )
    ferOp = electronic_energy_from_ints.second_q_ops()['ElectronicEnergy'] 
    #es_problem = ElectronicStructureProblem(electronic_energy_from_ints)
    qubit_converter = QubitConverter(mapper = JordanWignerMapper())
    qubitOp = qubit_converter.convert(ferOp) 

    #print(ferOp)
   
    ## eigval = eigvals(hgen) 
    ## idx = eigval.argsort()[::-1]   
    ## eigval = eigval[idx]
    ## print('fermionic eigenvalues', eigval)

    ## ham_mat = qubitOp.to_matrix()
    ## eigval = eigvals(ham_mat) 
    ## idx = eigval.argsort()[::-1]   
    ## eigval = eigval[idx]
    ## print('qubit eigenvalues', eigval)

    return qubitOp

def get_unitary_gate(pauli_op, gate_label):
    '''
    Get the final unitary transformation between quantum states from pauli_ops
    Note the -1 factor is for convention because HamiltonianGate(h) = exp(-1j*h*t)
    '''

    return HamiltonianGate(-pauli_op, time = 1, label=gate_label)

def get_Ulo2mo_from_loints(h1, h2, num_ele, gate_label=None):
    ''' 
    Given the 1- and 2-e integrals in LO basis
    return the unitary transformation from LO to MO
    '''

    U_orb_rot = get_mo_coeff(h1, h2, num_ele)
    h_orb_rot = get_hermitian_generator(U_orb_rot)
    pauli_sum_op = get_pauli_sum(h_orb_rot)
    U_lo2mo = get_unitary_gate(pauli_sum_op, gate_label)

    return U_lo2mo


# inverse QFT procedure
def qft_dagger(qc, register):
    ''' 
    Given a quantum circuit qc, and a quantum register for the evaluation qubits "register"
    implement the inverse QFT on the evaluation qubits
    '''
    
    n = register.size
    for qubit in range(n // 2):
        qc.swap(register[qubit], register[n - qubit - 1])
    for j in range(n):
        for m in range(j):
            qc.cp(-math.pi/float(2**(j-m)), register[m], register[j])
        qc.h(register[j])

def post_process_swap_counts(swap_result, f0_target, f1_target):
    '''
    input: simulation results from simulator for the swap test
           measurement results corresponds to the evaluation of frag 0
           measurement results corresponds to the evaluation of frag 1
    output: overlap Tr[\rho_A \rho_B]
    '''
    num_evaluation_qubits = int((len(swap_result[0])-1) / 2)
    total_counts = 0.0
    hit_counts = 0.0
    hit_counts_m1 = 0.0
    #print(num_evaluation_qubits)

    for count in swap_result:
        total_counts += 1
        # print(count)
        # print(count[num_evaluation_qubits:-2])
        if count[0:num_evaluation_qubits] == f1_target and count[num_evaluation_qubits:-1] == f0_target:
            hit_counts += 1
            hit_counts_m1 += float(count[-1])
    print('(qpe post-selection) hit_counts', hit_counts)
    print('SWAP measurement counts', hit_counts_m1)
    print(total_counts)

    overlap = hit_counts_m1 / hit_counts
    # print('Prob[M = 1] = %.4e' % overlap) 
    overlap = 1 - 2*overlap
    print('Overlap Tr[rho_A rho_B] = %.4e' % overlap) 

    return overlap

def get_index(array, element):
    '''
    return the index of the element in a descending sorted array
    '''

    counter = 0
    while (array[counter] - element > 1e-8):
        counter +=1
    return counter


def post_process_qpe_counts(mol_name, qpe_result, scale, id_coefficient, NumEvalQubits, num_shots):
    '''
    input: simulation results from simulator, and the Hamiltonain scale factor
    output: 
    '''

    answer = {}
    keyofenergy = {}
    counts = qpe_result.get_counts()
    for key in counts:
        scaledPhase = int(key, 2) / 2 ** NumEvalQubits
        if scaledPhase <= 0.5:
            #energy = scaledPhase * math.pi / scale.scale + id_coefficient * scale.scale
            energy = scaledPhase * 2* math.pi / scale.scale + id_coefficient
        else:
            #energy = (scaledPhase - 1) * math.pi / scale.scale + id_coefficient * scale.scale
            energy = (scaledPhase - 1)*  2*math.pi / scale.scale + id_coefficient
        answer[energy] = counts[key]
        keyofenergy[energy] = key
    
    energyResult = {}
    for energy in sorted(answer):
        if energy < 0:
            energyResult[energy] = answer[energy]
            print('{:.8f}'.format(round(energy, 8)) + ': ' + str(answer[energy]))
    
    # Sort energy by values to find most probable energy
    energy_estimate = max(answer, key = answer.get)
    print("Energy: " + str(energy_estimate) + ', key:' + keyofenergy[energy_estimate] + '\n')
    
    # compute variance of energies
    mean_energy = sum([value * answer[value] for value in answer.keys()]) / num_shots
    var_energy = sum([((value - mean_energy) ** 2) * answer[value] for value in answer.keys()])/num_shots
    print('Mean Energy: {}'.format(mean_energy) + '\n')
    print('Energy Variance: {}'.format(var_energy) + '\n')
   
    plt.plot(energyResult.keys(), energyResult.values())
    figname = mol_name + 'qpe_test_counts_energy.png'
    plt.savefig(figname, bbox_inches = 'tight', dpi=300)

    hist = plot_histogram(counts, title='QPE Testing Counts')
    hist.savefig(mol_name + 'qpe_test_counts_hist.png', bbox_inches = 'tight', dpi=300)

    return

def get_qpe_from_ints(mol_name, h1, Ulo2mo_gate, num_evaluation_qubits = 2, num_ele_up = 2, num_ele_down = 2, h2 = None, mapper = "JW", flag_test_qpe = False, flag_spin_separate = False):
    '''
    generate the QPE circuit given the 1 and 2 e integrals.
    without measurements.
    '''
    assert h1.all() != None
    nao = h1.shape[0]
    num_spin_orbitals = 2*nao 
    
    if mapper == "JW":
        qubit_converter = QubitConverter(mapper = JordanWignerMapper())

    electr_str = ElectronicEnergy.from_raw_integrals(ElectronicBasis.MO, h1, h2)
    ferOp = electr_str.second_q_ops()['ElectronicEnergy']
    # print(ferOp) 
    qubitOp = qubit_converter.convert(ferOp) # this will not give two-qubit reduction

    ## eigval = eigvals(hgen) 
    ## idx = eigval.argsort()[::-1]   
    ## eigval = eigval[idx]
    ## print('fermionic eigenvalues', eigval)

    ham_mat = qubitOp.to_matrix()
    eigval, eigvec = eigh(ham_mat) 
    #print('qubit eigenvalues', eigval)
    #print('eig vecs', eigvec)
    idx = eigval.argsort()[::-1]   
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]
    print('qubit eigenvalues', eigval)
    #print('eig vecs', eigvec)
    if (mol_name == 'H4_0d5A_'):
        gs_index = get_index(eigval.real,-9.60498729525741)
        print(eigval[gs_index])
        #print(eigvec[:,gs_index])
        if flag_spin_separate:
            rdm_up = qi.partial_trace(eigvec[:,gs_index], partial_trace_f0_up)
            rdm_dn = qi.partial_trace(eigvec[:,gs_index], partial_trace_f0_dn)
            #print(rdm_up)
            #print(rdm_dn)
            rdm = numpy.stack((rdm_up, rdm_dn))
            #print(rdm)
        else:
            # rdm = qi.partial_trace(eigvec[-1], [0,2])
            rdm = qi.partial_trace(eigvec[:,gs_index], partial_trace_f0)
            print(rdm)
    elif (mol_name == 'H4_0d81A_'):
        gs_index = get_index(eigval.real,-4.7060846185)
        print(eigval[gs_index])
        #print(eigvec[:,gs_index])
        if flag_spin_separate:
            rdm_up = qi.partial_trace(eigvec[:,gs_index], partial_trace_f1_up)
            rdm_dn = qi.partial_trace(eigvec[:,gs_index], partial_trace_f1_dn)
            #print(rdm_up)
            #print(rdm_dn)
            rdm = numpy.stack((rdm_up, rdm_dn))
            #print(rdm)
        else:
            #rdm = qi.partial_trace(eigvec[-1], [1,3])
            rdm = qi.partial_trace(eigvec[:,gs_index], partial_trace_f1)
            print(rdm)

    state_reg = QuantumRegister(num_spin_orbitals, name = 'system') # contains the initial state
    eval_reg = QuantumRegister(num_evaluation_qubits, name = 'evaluation') # contains the hpe evaluation qubits
    meas_reg = ClassicalRegister(num_evaluation_qubits, name = 'measurement')
    if flag_test_qpe:
        hpe = QuantumCircuit(eval_reg, state_reg, meas_reg, name='QPE')
    else:
        hpe = QuantumCircuit(eval_reg, state_reg, name='QPE')
    
    # initializes the hartree-fock state, note the lo2mo transformation has been used
    Ulo2mo_gate = get_Ulo2mo_from_loints(h1, h2, num_ele_up+num_ele_down, gate_label="U_lo2mo")
    Ulo2mo_gate_inv = Ulo2mo_gate.inverse()
    for iq in range(num_ele_up):
      hpe.x(state_reg[iq]) # block-spin format, 111000 111000
    for iq in range(num_ele_down):
      hpe.x(state_reg[iq+nao]) # block-spin format, 111000 111000
    hpe.append(Ulo2mo_gate_inv, state_reg)




    # construct the QPE circuit
    id_coefficient = 0.0
    ops = []
    #print(qubitOp.to_pauli_op())
    for op in qubitOp.to_pauli_op():
        p = op.primitive
        if p.x.any() or p.z.any():
            ops.append(op)
        else:
            id_coefficient += op.coeff
    qubitOp_noId = SummedOp(ops)
    # print(qubitOp_noId)
    
    print("Identity Coefficient: " + str(id_coefficient) + "\n")
    
    # hamiltonian scaling to generate the unitary operation
    scale = PhaseEstimationScale.from_pauli_sum(qubitOp_noId.to_pauli_op())
    #qubitOp_scaled = scale.scale * qubitOp.to_pauli_op()
    qubitOp_scaled = -qubitOp_noId.to_pauli_op() * scale.scale
    
    #scale = 0.5
    #qubitOp_scaled = qubitOp_noId.to_pauli_op() * scale
    
    unitaryOp = PauliTrotterEvolution().convert(qubitOp_scaled.exp_i())
    
    print("Scale: " + str(scale.scale) + "\n")
    

    # create the circuit for the unitary operation
    unitaryCircuit = unitaryOp.to_circuit().decompose()
    
    # constructs the HPE circuit
    hpe.h(eval_reg)
    for count in range(num_evaluation_qubits):
        hpe.compose(unitaryCircuit.power(2**count).control(), qubits = [eval_reg[count]] + state_reg[:], inplace = True)
    
    qft_dagger(hpe, eval_reg)
    


    if flag_test_qpe and flag_sim_qpe:
        hpe.measure(eval_reg, meas_reg)
        num_shots = 1024
        backend = Aer.get_backend(SimulatorType)
        quantum_instance = QuantumInstance(backend, shots = num_shots)
        
        qpe_test_result = quantum_instance.execute(hpe)
        post_process_qpe_counts(mol_name, qpe_test_result, scale, id_coefficient, num_evaluation_qubits, num_shots)



    hpe.draw(output='mpl', filename = mol_name + 'hpe_circ.png', reverse_bits = False)
    sys.stdout.flush()

    return hpe, scale, id_coefficient, rdm


def read_ints_from_file(mol_name, read_ints_basis = 'MO'):
    ''' 
    Read the 1-e integral from file
    '''
    index1e = 'i0f0'
    index2e = 'f0'
    print('***' + index1e + '***')
    
    if read_ints_basis == 'LO':
      r1 = h5py.File('integrals/' + mol_name + 'lo_h1.h5','r')
      r2 = h5py.File('integrals/' + mol_name + 'lo_eri_file.h5','r')
    elif read_ints_basis == 'MO':
      r1 = h5py.File('integrals/' + mol_name + 'h1.h5','r')
      r2 = h5py.File('integrals/' + mol_name + 'eri_file.h5','r')
    h1 = numpy.array(r1.get(index1e))
    h2 = numpy.array(r2.get(index2e))
    symm = 1 # change to 1 for no symmetry
    NumOrbitals = h1.shape[0]
    eri = ao2mo.restore(symm, h2, NumOrbitals)
    print(h1)
    # print(eri)
    r1.close()
    r2.close()

    return h1, eri

def get_overlap(rdm0, rdm1):

    overlap = numpy.trace(numpy.array(rdm0) @ numpy.array(rdm1))

    return overlap



## Main algorithm #############################################################
start_time = datetime.now()

# SCF (0.9 A) = -7.31629735125644
# SCF (0.5 A) = -9.60498729525741
# SCF (0.6 A) = -8.91454076964582
# SCF (1A) = -6.90283990824471
mol_name_f0 = 'H4_0d5A_'
mol_name_f1 = 'H4_0d5A_'
ints_basis = 'LO'
num_evaluation_qubits = 5
f0_target_energy = '11011'   # h4, 0.5 A
f1_target_energy = '11011'   # h4, 0.5 A
#f0_target_energy = '11101'   # h4, 0.8 A
#f1_target_energy = '11101'   # h4, 0.8 A
num_spin_orbitals = 8
num_spatial_orbitals = int(num_spin_orbitals / 2)
num_ele_up = 2
num_ele_down = 2
flag_test_qpe = False
flag_sim_qpe = True
flag_spin_separate = False
max_ovlp_orb_num = 1

#partial_trace_f0 = [0,1,2,3,4,5,6]
if max_ovlp_orb_num < num_spatial_orbitals:
    partial_trace_f0 = list(range(max_ovlp_orb_num,num_spatial_orbitals)) + list(range(max_ovlp_orb_num+num_spatial_orbitals,2*num_spatial_orbitals))
else:
    partial_trace_f0 = []

# [2,3, 6,7]
print(partial_trace_f0)
partial_trace_f0_up = [0,1,3,4,5]
partial_trace_f0_dn = [0,1,2,3,4]

#partial_trace_f1 = [1,2,4,5]
partial_trace_f1 = partial_trace_f0
partial_trace_f1_up = [1,2,3,4,5]
partial_trace_f1_dn = [0,1,2,4,5]

SimulatorType = "aer_simulator"

# specify the overlapping region
num_overlap_orbitals = 2
f0_orb2qubit = [2] # list of orbitals in frag 0 that overlaps with frag 1 
f1_orb2qubit = [0] # list of orbitals in frag 1 that overlaps with frag 0


## Read in the integrals from files
h1_f0, h2_f0 = read_ints_from_file(mol_name_f0, read_ints_basis = ints_basis)
h1_f1, h2_f1 = read_ints_from_file(mol_name_f1, read_ints_basis = ints_basis)
h2_f0 = numpy.zeros((num_spatial_orbitals, num_spatial_orbitals, num_spatial_orbitals, num_spatial_orbitals))
h2_f1 = numpy.zeros((num_spatial_orbitals, num_spatial_orbitals, num_spatial_orbitals, num_spatial_orbitals))

## Get the Ulo2mo_gate transformation. Note the same basis has to be used for both fragments. Here we will use the Ulo2mo_gate obtained from f0 to both fragments.
# h2_f0 = numpy.zeros((num_spatial_orbitals, num_spatial_orbitals, num_spatial_orbitals, num_spatial_orbitals))
Ulo2mo_gate = get_Ulo2mo_from_loints(h1_f0, h2_f0, num_ele_up+num_ele_down, gate_label="U_lo2mo")
Ulo2mo_gate2 = get_Ulo2mo_from_loints(h1_f1, h2_f1, num_ele_up+num_ele_down, gate_label="U_lo2mo2")

## Get the Two QPE circuit
# if flag_spin_separate:
hpe_f0, scale_f0, id_coeff_f0, rdm0 = get_qpe_from_ints(mol_name_f0, h1_f0, Ulo2mo_gate, num_evaluation_qubits = num_evaluation_qubits, num_ele_up = num_ele_up, num_ele_down = num_ele_down, h2 = h2_f0, flag_test_qpe = flag_test_qpe, flag_spin_separate=flag_spin_separate)

hpe_f1, scale_f1, id_coeff_f1, rdm1 = get_qpe_from_ints(mol_name_f1, h1_f1, Ulo2mo_gate2, num_evaluation_qubits = num_evaluation_qubits, num_ele_up = num_ele_up, num_ele_down = num_ele_down, h2 = h2_f1, flag_test_qpe = flag_test_qpe, flag_spin_separate=flag_spin_separate)
# else:
#     hpe_f0, scale_f0, id_coeff_f0, rdm0 = get_qpe_from_ints(mol_name_f0, h1_f0, Ulo2mo_gate, num_evaluation_qubits = num_evaluation_qubits, num_ele_up = num_ele_up, num_ele_down = num_ele_down, flag_test_qpe = flag_test_qpe, flag_spin_separate=flag_spin_separate)
#     hpe_f1, scale_f1, id_coeff_f1, rdm1 = get_qpe_from_ints(mol_name_f1, h1_f1, Ulo2mo_gate, num_evaluation_qubits = num_evaluation_qubits, num_ele_up = num_ele_up, num_ele_down = num_ele_down, flag_test_qpe = flag_test_qpe, flag_spin_separate=flag_spin_separate)

# exact_overlap = (get_overlap(rdm0_up, rdm1_up) + get_overlap(rdm0_dn, rdm1_dn)
# print(rdm0)
# print(rdm1)
if flag_spin_separate:
    exact_overlap = (get_overlap(rdm0[0], rdm1[0]) + get_overlap(rdm0[1], rdm1[1]))/2.0 
else:
    exact_overlap = get_overlap(rdm0, rdm1)
print("exact overlap = %.6e" % exact_overlap)


if not flag_test_qpe:
    ## Compose together with SWAP test 
    qr_anc = QuantumRegister(1, 'anc')
    qr0 = QuantumRegister(num_spin_orbitals+num_evaluation_qubits, 'f0_QPE')
    qr1 = QuantumRegister(num_spin_orbitals+num_evaluation_qubits, 'f1_QPE')
    cr = ClassicalRegister(2*num_evaluation_qubits+1, 'measure')
    
    circ = QuantumCircuit(qr_anc, qr0, qr1, cr, name="Quantum Matching")
    circ.append(hpe_f0, qr0)
    circ.append(hpe_f1, qr1)
   
    circ.h(0) 
    #for iq in range(num_overlap_orbitals):
    #    circ.cswap(qr_anc, qr0[num_evaluation_qubits+f0_orb2qubit[iq]], qr1[num_evaluation_qubits+f1_orb2qubit[iq]]) # spin up qubits
    #    circ.cswap(qr_anc, qr0[num_evaluation_qubits+int(f0_orb2qubit[iq]+num_spin_orbitals/2)], qr1[num_evaluation_qubits+int(f1_orb2qubit[iq]+num_spin_orbitals/2)]) # spin down qubits
    for iq in range(max_ovlp_orb_num):
        circ.cswap(qr_anc, qr0[num_evaluation_qubits+iq], qr1[num_evaluation_qubits+iq]) # spin up qubits
        circ.cswap(qr_anc, qr0[num_evaluation_qubits+num_spatial_orbitals+iq], qr1[num_evaluation_qubits+num_spatial_orbitals+iq]) # spin down qubits
    circ.h(0)
    
    circ.barrier()
    
    ## Now do the measurement
    circ.measure([0] + list(range(1,num_evaluation_qubits+1)) + list(range(1+num_spin_orbitals+num_evaluation_qubits, 1+2*num_evaluation_qubits+num_spin_orbitals)), cr)
    # circ.measure([0] + list(range(1,num_evaluation_qubits+1)), cr[0:num_evaluation_qubits+1])
    
    
    circ.draw(output='mpl', filename = 'qpe+swap_circ.png', reverse_bits = False)
    
    
    
    # ----------- comment out the simulation part for now ---------------
    ## Now simulate
    
    # --------- coment out the FakeMumbai transpile ------------ 
    # backend = FakeMumbai()
    # quantum_instance = QuantumInstance(backend)
    # shots = 1024
    # for itrans in range(5):
    #     preprocessing_time = datetime.now()
    #     t_circ = transpile(circ, backend)
    #     transpile_time = datetime.now()
    #     print('Transpile # {}:'.format(itrans))
    #     print("Transpiled Depth: " + str(t_circ.depth()))
    #     #print("Transpiled Depth (filter): " + str(t_hpe.depth(filter_function=('rx','ry','rz'))))
    #     print('Transpile Duration: {}'.format(transpile_time - preprocessing_time) + "\n")
    # --------- coment out the FakeMumbai transpile ------------ 
    
    # Transpile for simulator
    simulator = Aer.get_backend('aer_simulator')
    t_circ = transpile(circ, simulator)
    
    # # run and get counts
    # result = simulator.run(t_circ).result()
    # counts = result.get_counts(t_circ)
    # hist = plot_histogram(counts, title='QPE+Quantum Matching Counts')
    # hist.savefig('quantum-matching.png', bbox_inches = 'tight', dpi=300)
    
   
    simulation_results_filename = 'h4_sim_ovlp1.txt' 
    simulation_results_file = open(simulation_results_filename, 'w')
    nblocks = 10
    for ib in range(nblocks):
        # run and get memory
        # the output measurement bits convention: lsb = bit 0 (last digit)
        shots = 1e3
        result = simulator.run(t_circ, shots=shots, memory=True).result()
        memory = result.get_memory(t_circ)
        #print(memory)
        
        # now post-selection on the memory
        iovlp = post_process_swap_counts(memory, f0_target_energy, f1_target_energy)
        simulation_results_file.write("%8d\t%12.8e\n" % (ib, iovlp))
        sys.stdout.flush()
    simulation_results_file.close()
        
