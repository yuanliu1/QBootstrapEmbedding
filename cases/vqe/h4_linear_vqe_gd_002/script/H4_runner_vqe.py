"""
Script to run QBE VQE GD solve with linear constraints on H4 chain
"""

# Imports
import os, sys, types
import pathlib
import argparse
import shutil

import pickle
import copy

import numpy as np
from matplotlib import pyplot as plt
import qiskit.quantum_info as qi
import h5py

from qiskit import QuantumCircuit
from qiskit import Aer

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.circuit.library import HartreeFock, UCCSD

# Local package imports
# Update this with setup & develop later
PROJECT_PATH = str(pathlib.Path().resolve().parent)
sys.path.append(PROJECT_PATH)

import qbe


# Function for reading hamiltonian data
def init_fragment_hamiltonians(fragments, hamiltonian_data):
    '''
    initialize a dictionary where the keys are fragment labels
    and the values are the corresponding fragment Hamiltonian matrices
    '''
    fragment_hamiltonians = {}
    for i in range(len(fragments)):
        fragment_hamiltonians[str(i)] = np.matrix(hamiltonian_data['i0f'+str(i)])
    return fragment_hamiltonians


# Parse argument inputs to file
parser = argparse.ArgumentParser(description='VQE GD Run on H4')
parser.add_argument('--type_constraints', type=str, default='linear', metavar='type_constraints_qbe')
parser.add_argument('--expt_id', type=int, default=1, metavar='N')
parser.add_argument('--run_seed', type=int, default=0, metavar='N')
args = parser.parse_args()

# Load the Hamiltonian matrices for each fragment
DATA_dir = '../data/'
hamiltonian_file = 'H4_initial_ham.h5'
molecule_name = 'H4'

hamiltonian_data = h5py.File(DATA_dir + hamiltonian_file, 'r')

print('Loaded Hamiltonian from %s for molecule %s' % (hamiltonian_file, molecule_name))

# fragment info
n_frags = 2
n_qubits = 8

labels_fragments = ['0', '1']
fragment_info = {'0': [0,1,2,4,5,6],'1': [0,1,2,4,5,6]}
fragment_nb = {'0': {'1': {'n_sites': 2, 'edge': [(2,1),(6,5)], 'center': [(1,0),(5,4)]}},
                    '1': {'0': {'n_sites': 2, 'edge': [(0,1),(4,5)], 'center': [(1,2),(5,6)]}}}

print('Fragment info: n_frags=%d, n_qubits (per frag)=%d' % (n_frags, n_qubits))

# load in fragment hamiltonians
fragment_init = init_fragment_hamiltonians(fragment_info, hamiltonian_data)

# create fragment object
qbe_frag_init = qbe.fragment_hamiltonian.qbe_fragment_qubit(n_frags, n_qubits, labels_fragments,
                                                            fragment_info, fragment_nb, fragment_init)

# For saving and logging info
expt_number = args.expt_id
type_constraints = args.type_constraints
type_gs_solver = 'vqe'
FLAG_logger = True

# Creation of save directory and summary log-file
SAVE_DIR = molecule_name.lower() + '_' + type_constraints + '_vqe_gd_%03d' % expt_number
log_filename = SAVE_DIR + '/log_job_%d.txt' % expt_number

if not os.access(SAVE_DIR, os.F_OK):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Create summary log-file and denote entries
    f_log = open(log_filename, "a+")
    f_log.write("iter eig_calls avg_grad_value rmse_fragment_rho runtime \n")
    f_log.close()

# Save log-files of each run to different folder
SAVE_DIR_runs = SAVE_DIR + '/log_runs'

if not os.access(SAVE_DIR_runs, os.F_OK):
    os.makedirs(SAVE_DIR_runs, exist_ok=True)

# Save script to folder to know what we ran
SAVE_DIR_script = SAVE_DIR + '/script'

if not os.access(SAVE_DIR_script, os.F_OK):
    os.makedirs(SAVE_DIR_script, exist_ok=True)

current_script_file = os.path.basename(__file__)
shutil.copy(current_script_file, SAVE_DIR_script)

run_log_filename = SAVE_DIR_runs + '/log_%d.txt' % args.run_seed

# Define Solve
qbe_frag = copy.deepcopy(qbe_frag_init)

optimizer_options = {'max_iters': 20, 'LR_init': 0.1,
                     'LR_schedule': None,
                     'n_gd_iters': 10, 'THRES_GRAD': 1e-9}
nshots = 100000

# Setup VQE Solve
# Create an initial state -- HF state
qubit_converter = QubitConverter(mapper=JordanWignerMapper())

num_alpha = 2
num_beta = 2
num_particles = (num_alpha, num_beta)

num_molecular_orbitals = 4
num_spin_orbitals = 2*num_molecular_orbitals

hf_state = HartreeFock(num_spin_orbitals, num_particles, qubit_converter)

# ansatz
ansatz = UCCSD(qubit_converter, num_particles, num_spin_orbitals, reps=1, initial_state=hf_state)

# LO2MO Transformation
U_lo2mo_H4 = qbe.utils.lo2mo_trans.get_unitary_lo2mo(file_name_1e='../data/H4_pert_h1.h5',
                                                     file_name_2e='../data/H4_pert_eri_file.h5',
                                                     num_ele=4)
U_lo2mo_H4 = np.array(U_lo2mo_H4)

# backend
backend = Aer.get_backend('statevector_simulator')

# Initialize QBEVQESolver
vqe_run_seed = 10 * (args.run_seed + 2)
vqe_solver = qbe.QBEVQESolver(n_qubits=8, ansatz=ansatz, backend=backend, seed=vqe_run_seed, nshots=nshots,
                              FLAG_lo2mo=True, U_lo2mo=U_lo2mo_H4)

qbe_solver_lin_vqe = qbe.quantum_bootstrap.qbe_solver_qubit(qbe_frag, type_constraint='linear',
                                                            type_gs_solver='vqe',
                                                            gs_solver=vqe_solver,
                                                            optimizer_options=optimizer_options)

# Solve!
print('Started QBE-VQE GD solve with type_constraints=%s max_iters=%d, n_gd_iters=%d' % (type_constraints,
                                                                                 optimizer_options['max_iters'],
                                                                                 optimizer_options['n_gd_iters']))

ds_vqe = qbe_solver_lin_vqe.gd_solve(FLAG_verbose=True, FLAG_logger=True, log_filename=run_log_filename)

# Update RMSE and log results
f_log = open(log_filename, "a+")

n_iter = ds_vqe['iterations']
norm_gradients = ds_vqe['norm_gradients']
rmse_error = ds_vqe['rmse_error_fragment_rho']
n_eig_calls = ds_vqe['n_eig_calls']
run_time = ds_vqe['run_time']

for i_iter in range(optimizer_options['max_iters']):
    f_log.write("%d %d %3.18f %3.18f %f \n" % (n_iter[i_iter], n_eig_calls[i_iter],
                                               norm_gradients[i_iter], rmse_error[i_iter],
                                               run_time[i_iter]))

f_log.close()

# pickle_result_file = SAVE_DIR + '/Run_%d.pickle' % i_run
# with open(pickle_result_file, 'wb') as handle:
#     pickle.dump(ds_vqe, handle, protocol=pickle.HIGHEST_PROTOCOL)