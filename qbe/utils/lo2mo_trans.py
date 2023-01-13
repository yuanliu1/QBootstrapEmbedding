# 07/21/2022, generate the unitary transformation in quantum states from LO representation to MO representation. 

import numpy as np
import h5py
from pyscf import ao2mo, gto, scf
from functools import reduce
from scipy.linalg import logm, expm, block_diag, eigvals
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit.extensions import HamiltonianGate
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicEnergy,
)


def get_mo_coeff(fn_1e, fn_2e, nele = 2):
    '''
    Read in the 1e- and 2e-integrals in LO and then generate 
    the orbital rotation unitary matrix to go from LOs to MOs.

    inputs: file names for 1e- and 2e integrals, number of electrons of the molecule
    output: the orbital rotation unitary matrix from LOs to MOs
    '''

    index1e = 'i0f0'
    index2e = 'f0'

    # access 1e- integral 
    r = h5py.File(fn_1e,'r')
    h1 = np.array(r.get(index1e))
    #print(h1)
    r.close()
    
    # access 2e- integral for fragment 3
    r = h5py.File(fn_2e,'r')
    h2 = np.array(r.get(index2e))
    #print(h2)
    r.close()
    
    symm = 8 # change to 8 for 8-fold symmetry
    nao = h1.shape[0]
    eri = ao2mo.restore(symm, h2, nao)
    
    mol = gto.M()
    mol.nelectron = nele
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(nao)
    mf._eri = eri
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    
    mf.kernel(max_cycle = 1)
    #mf.kernel()

    # check if mo_coeff is indeed a unitary
    c = mf.mo_coeff
    assert np.allclose(reduce(np.dot, (c.T, c)), np.eye(nao))

    return block_diag(c.T, c.T)


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
        ElectronicBasis.SO, hgen, np.zeros((nao, nao, nao, nao))
    )
    ferOp = electronic_energy_from_ints.second_q_ops()[0]  
    #es_problem = ElectronicStructureProblem(electronic_energy_from_ints)
    qubit_converter = QubitConverter(mapper = JordanWignerMapper())
    qubitOp = qubit_converter.convert(ferOp) 

    print(ferOp)
   
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


def get_unitary_gate(pauli_op):
    '''
    Get the final unitary transformation between quantum states from pauli_ops
    Note the -1 factor is for convention because HamiltonianGate(h) = exp(-1j*h*t)
    '''

    return HamiltonianGate(-pauli_op, time=1)


def get_unitary_lo2mo(file_name_1e='../data/H4_h1.h5', file_name_2e='../data/H4_eri_file.h5', num_ele=4):
    U_orb_rot = get_mo_coeff(file_name_1e, file_name_2e, num_ele)
    h_orb_rot = get_hermitian_generator(U_orb_rot)
    pauli_sum_op = get_pauli_sum(h_orb_rot)
    U_lo2mo = get_unitary_gate(pauli_sum_op)

    return U_lo2mo


# ############### Main Algorithm ########
# file_name_1e = 'data/H4_h1.h5'
# file_name_2e = 'data/H4_eri_file.h5'
# num_ele = 4
#
# U_orb_rot = get_mo_coeff(file_name_1e, file_name_2e, num_ele)
# h_orb_rot = get_hermitian_generator(U_orb_rot)
# pauli_sum_op = get_pauli_sum(h_orb_rot)
# U_lo2mo = get_unitary_gate(pauli_sum_op)
#
# print(U_orb_rot)
# print(h_orb_rot)
# print(pauli_sum_op)
# print(U_lo2mo)





################## (For Testing) Output #####################
#/usr/local/anaconda3/envs/env_qiskit/lib/python3.9/site-packages/pyscf/lib/misc.py:46: H5pyDeprecationWarning: Using default_file_mode other than 'r' is deprecated. Pass the mode to h5py.File() instead.
#  h5py.get_config().default_file_mode = 'a'
#Overwritten attributes  get_hcore get_ovlp  of <class 'pyscf.scf.hf.RHF'>
#converged SCF energy = -4.39164718431802
#Fermionic Operator
#register length=8, number terms=32
#  (-0.667278480957818-0.00606785836270474j) * ( +_0 -_1 )
#+ (-1.1831921500114544+1.3774187893219656e-15j) * ( +_0 -_2 )
#+ (-0.0082016110955912+0.49367755478061776j) * ( +_0 -_3 )
#+ (0.667278480957 ...
#[[ 0.39379907  0.59680244  0.58730085 -0.3792451   0.          0.
#   0.          0.        ]
# [ 0.58730085  0.3792451  -0.39379907  0.59680244  0.          0.
#   0.          0.        ]
# [ 0.58730085 -0.3792451  -0.39379907 -0.59680244  0.          0.
#   0.          0.        ]
# [ 0.39379907 -0.59680244  0.58730085  0.3792451   0.          0.
#   0.          0.        ]
# [ 0.          0.          0.          0.          0.39379907  0.59680244
#   0.58730085 -0.3792451 ]
# [ 0.          0.          0.          0.          0.58730085  0.3792451
#  -0.39379907  0.59680244]
# [ 0.          0.          0.          0.          0.58730085 -0.3792451
#  -0.39379907 -0.59680244]
# [ 0.          0.          0.          0.          0.39379907 -0.59680244
#   0.58730085  0.3792451 ]]
#[[ 7.82044402e-01+9.16042209e-16j -6.67278481e-01-6.06785836e-03j
#  -1.18319215e+00+1.37741879e-15j -8.20161110e-03+4.93677555e-01j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j]
# [-6.67278481e-01+6.06785836e-03j  5.69354592e-01+1.36640193e-15j
#   1.00955733e+00+9.29438619e-03j  6.99801518e-03-7.62253908e-01j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j]
# [-1.18319215e+00+9.40337435e-16j  1.00955733e+00-9.29438619e-03j
#   1.79010765e+00-8.23816382e-17j  1.24086073e-02+7.56186050e-01j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j]
# [-8.20161110e-03-4.93677555e-01j  6.99801518e-03+7.62253908e-01j
#   1.24086073e-02-7.56186050e-01j  8.60135619e-05+9.62528598e-16j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j]
# [ 0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#   7.82044402e-01+9.22981103e-16j -6.67278481e-01-6.06785836e-03j
#  -1.18319215e+00+1.39471767e-15j -8.20161110e-03+4.93677555e-01j]
# [ 0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#  -6.67278481e-01+6.06785836e-03j  5.69354592e-01+1.35493950e-15j
#   1.00955733e+00+9.29438619e-03j  6.99801518e-03-7.62253908e-01j]
# [ 0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#  -1.18319215e+00+9.40337435e-16j  1.00955733e+00-9.29438619e-03j
#   1.79010765e+00-8.32667268e-17j  1.24086073e-02+7.56186050e-01j]
# [ 0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#   0.00000000e+00+0.00000000e+00j  0.00000000e+00+0.00000000e+00j
#  -8.20161110e-03-4.93677555e-01j  6.99801518e-03+7.62253908e-01j
#   1.24086073e-02-7.56186050e-01j  8.60135619e-05+9.99200722e-16j]]
#(3.141592653589795+3.1782228437587637e-15j) * IIIIIIII
#+ (-4.300678093874244e-05-4.996003610813204e-16j) * ZIIIIIII
#+ (-0.8950538231896502+4.163336342344337e-17j) * IZIIIIII
#+ (-0.28467729581861034-6.774697477951776e-16j) * IIZIIIII
#+ (-0.39102220100569807-4.614905515526303e-16j) * IIIZIIII
#+ (-4.3006780938770195e-05-4.812642987623557e-16j) * IIIIZIII
#+ (-0.8950538231896502+4.1190819115499334e-17j) * IIIIIZII
#+ (-0.28467729581861034-6.832009625055455e-16j) * IIIIIIZI
#+ (-0.39102220100569807-4.580211046006767e-16j) * IIIIIIIZ
#+ (0.006204303643624204+1.1102230246251565e-16j) * XXIIIIII
#+ (-0.37809302488671803-2.7755575615628914e-17j) * YXIIIIII
#+ (0.37809302488671803+2.7755575615628914e-17j) * XYIIIIII
#+ (0.006204303643624204+1.1102230246251565e-16j) * YYIIIIII
#+ (0.003499007587802666-2.7755575615628914e-17j) * XZXIIIII
#+ (0.38112695406807084-2.7755575615628914e-17j) * YZXIIIII
#+ (-0.38112695406807084+2.7755575615628914e-17j) * XZYIIIII
#+ (0.003499007587802666-2.7755575615628914e-17j) * YZYIIIII
#+ (0.504778666345695-5.555451931815725e-16j) * IXXIIIII
#+ (-0.004647193093392482-1.6653345369377348e-16j) * IYXIIIII
#+ (0.004647193093392482+1.6653345369377348e-16j) * IXYIIIII
#+ (0.504778666345695-5.555451931815725e-16j) * IYYIIIII
#+ (-0.004100805547795641+2.7755575615628914e-17j) * XZZXIIII
#+ (-0.24683877739030888+2.7755575615628914e-17j) * YZZXIIII
#+ (0.24683877739030888-2.7755575615628914e-17j) * XZZYIIII
#+ (-0.004100805547795641+2.7755575615628914e-17j) * YZZYIIII
#+ (-0.5915960750057272+5.837637774408622e-16j) * IXZXIIII
#+ (-0.5915960750057272+5.837637774408622e-16j) * IYZYIIII
#+ (-0.33363924047890897+3.37403716077489e-16j) * IIXXIIII
#+ (0.003033929181352711-5.551115123125783e-17j) * IIYXIIII
#+ (-0.003033929181352711+5.551115123125783e-17j) * IIXYIIII
#+ (-0.33363924047890897+3.37403716077489e-16j) * IIYYIIII
#+ (0.006204303643624204+1.1102230246251565e-16j) * IIIIXXII
#+ (-0.37809302488671803-2.7755575615628914e-17j) * IIIIYXII
#+ (0.37809302488671803+2.7755575615628914e-17j) * IIIIXYII
#+ (0.006204303643624204+1.1102230246251565e-16j) * IIIIYYII
#+ (0.003499007587802666-5.551115123125783e-17j) * IIIIXZXI
#+ (0.3811269540680708-2.7755575615628914e-17j) * IIIIYZXI
#+ (-0.3811269540680708+2.7755575615628914e-17j) * IIIIXZYI
#+ (0.003499007587802666-5.551115123125783e-17j) * IIIIYZYI
#+ (0.504778666345695-5.616167253474913e-16j) * IIIIIXXI
#+ (-0.004647193093392477-1.6653345369377348e-16j) * IIIIIYXI
#+ (0.004647193093392477+1.6653345369377348e-16j) * IIIIIXYI
#+ (0.504778666345695-5.616167253474913e-16j) * IIIIIYYI
#+ (-0.004100805547795634+1.3877787807814457e-17j) * IIIIXZZX
#+ (-0.24683877739030885+3.469446951953614e-17j) * IIIIYZZX
#+ (0.24683877739030885-3.469446951953614e-17j) * IIIIXZZY
#+ (-0.004100805547795634+1.3877787807814457e-17j) * IIIIYZZY
#+ (-0.5915960750057272+5.794390561000153e-16j) * IIIIIXZX
#+ (-0.5915960750057272+5.794390561000153e-16j) * IIIIIYZY
#+ (-0.33363924047890897+3.371868756429919e-16j) * IIIIIIXX
#+ (0.0030339291813527075-5.551115123125783e-17j) * IIIIIIYX
#+ (-0.0030339291813527075+5.551115123125783e-17j) * IIIIIIXY
#+ (-0.33363924047890897+3.371868756429919e-16j) * IIIIIIYY
#Instruction(name='hamiltonian', num_qubits=8, num_clbits=0, params=[array([[-3.33066907e-16-3.94430453e-31j,  0.00000000e+00+0.00000000e+00j,
#         0.00000000e+00+0.00000000e+00j, ...,
#         0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
#         0.00000000e+00+0.00000000e+00j],
#       [ 0.00000000e+00+0.00000000e+00j, -7.82044402e-01-9.16042209e-16j,
#         6.67278481e-01+6.06785836e-03j, ...,
#         0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
#         0.00000000e+00+0.00000000e+00j],
#       [ 0.00000000e+00+0.00000000e+00j,  6.67278481e-01-6.06785836e-03j,
#        -5.69354592e-01-1.36640193e-15j, ...,
#         0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
#         0.00000000e+00+0.00000000e+00j],
#       ...,
#       [ 0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
#         0.00000000e+00+0.00000000e+00j, ...,
#        -5.71383072e+00-4.99004376e-15j,  6.67278481e-01+6.06785836e-03j,
#         0.00000000e+00+0.00000000e+00j],
#       [ 0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
#         0.00000000e+00+0.00000000e+00j, ...,
#         6.67278481e-01-6.06785836e-03j, -5.50114091e+00-5.44040348e-15j,
#         0.00000000e+00+0.00000000e+00j],
#       [ 0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
#         0.00000000e+00+0.00000000e+00j, ...,
#         0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
#        -6.28318531e+00-6.35644569e-15j]]), 1])


