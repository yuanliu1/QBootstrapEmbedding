# Compute the energy from saved gs vector for each run. 12/12/2022 

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
import qiskit.quantum_info as qi
import h5py
import copy
import matplotlib
import scipy.sparse
import pickle
#from timeit import default_timer as timer


run = 12
num_qubits = 12 # number of qubits in the Hamiltonian, i.e. the Hamiltonian matrix has dimension (2**num_qubits)
num_qubits_full = 16
# keys: fragment labels
# values: list of indices of qubits in the fragment
# example: {'0': [0,1,2,6,7,8]} means that fragment '0' contains qubits with indices 0, 1, 2, 6, 7, and 8
fragments = {'0': [0,1,2,6,7,8],'1': [0,1,2,6,7,8],'2': [0,1,2,6,7,8],
             '3': [0,1,2,6,7,8], '4': [0,1,2,6,7,8], '5': [0,1,2,6,7,8]} 

# Let fragment A be the main fragment, then let B be any fragment which overlaps with A
# keys: label of fragment A
# values: dict, with
    # keys: label of fragment B
    # values: list of ordered qubit pairs on fragments A and B which overlap

# example: {'0': {'1': [[2,1],[8,7]]}} means that fragment '0' and fragment '1' overlap at two sites: qubit 2 on
    # fragment A overlaps with qubit 1 on fragment B; also, qubit 8 on fragment A overlaps with qubit 7 on fragment B.
fragment_overlap = {'0': {'1': [[2,1],[8,7]]}, 
                    '1': {'0': [[0,1],[6,7]], '2': [[2,1],[8,7]]},
                    '2': {'1': [[0,1],[6,7]], '3': [[2,1],[8,7]]},
                    '3': {'2': [[0,1],[6,7]], '4': [[2,1],[8,7]]},
                    '4': {'3': [[0,1],[6,7]], '5': [[2,1],[8,7]]}, 
                    '5': {'4': [[0,1],[6,7]]}}



center_sites = {'0':[0,1,6,7], '1':[1,7],
               '2':[1,7], '3':[1,7],
               '4':[1,7], '5':[1,2,7,8],}

def init_fragment_hamiltonians():
    '''
    initialize a dictionary where the keys are fragment labels 
    and the values are the corresponding fragment Hamiltonian matrices
    '''
    fragment_hamiltonians = {}
    for i in range(len(fragments)):
        fragment_hamiltonians[str(i)] = np.array(hamiltonian_data['i0f'+str(i)])
    return fragment_hamiltonians


################### Load the Hamiltonian matrices for each fragment
hamiltonian_data = h5py.File('../../data/H8_initial_ham.h5', 'r')
fragment_hamiltonians = init_fragment_hamiltonians()
hamiltonian_data.close()

##################### Load ground state data #################
for irun in range(run):
  gs_file_name = 'gs_vec_lin_run' + str(irun) + '.pkl'
  print('Run ', irun)
  with open(gs_file_name, 'rb') as f:
    gs_vec = pickle.load(f)
    #print(list(gs_vec.keys()))
    
    sum_eng = 0.0
    # loop over fragments
    for frag in fragments: 
      # print(gs_vec[frag])
      tmp_energy = np.conjugate(gs_vec[frag].transpose()) @ (fragment_hamiltonians[frag]@gs_vec[frag])
      print('frag '+frag, ' energy is: ', tmp_energy)
      sum_eng += tmp_energy 

    f.close()
    print('Total energy for this run is', sum_eng)


