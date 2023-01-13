# Bug fixed on index of H8 labels. 11/13/2022
# Save the full density matrix of each fragment for each BE iteration. 12/11/2022

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

##################### Load preliminary data #################
with open('H8_HCAs.pkl', 'rb') as f:
    H_CAs = pickle.load(f)
target_energy = np.load('H8_target_energy.npy')[0]
target_occupation_number = 8
##################### Initialize ############################

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

# # load the Hamiltonian matrices for each fragment
hamiltonian_data = h5py.File('../../data/H8_initial_ham.h5', 'r')

##unfragmented_hamiltonian = scipy.sparse.load_npz('H8_1A_lo_initial_ham_sp.npz')


def get_ground_state(H): 
    '''
    returns ground state energy and statevector of a Hamiltonian matrix
    '''
    H = scipy.sparse.csr_matrix(H)
    eigvals, eigvec = scipy.sparse.linalg.eigsh(H, which = 'SA')
    min_eigval_index = np.argmin(eigvals)
    ground_state_energy = eigvals[min_eigval_index]
    ground_state = eigvec[:,min_eigval_index]
    return ground_state_energy, ground_state

def update_fragment_gs(fragment_hamiltonians): 
    '''
    updates the fragment ground state energies and fragment ground state density matrices
    '''
    fragment_gs_energies = {} # dictionary. keys: fragment labels, values: energy of fragment ground state
    fragment_gs = {} # dictionary. keys: fragment labels, values: fragment ground state density matrices
    for f in fragment_hamiltonians:
        gs_energy, gs = get_ground_state(np.array(fragment_hamiltonians[f]))
        fragment_gs_energies[f] = gs_energy
        fragment_gs[f] = gs
    return fragment_gs_energies, fragment_gs

def init_fragment_hamiltonians():
    '''
    initialize a dictionary where the keys are fragment labels 
    and the values are the corresponding fragment Hamiltonian matrices
    '''
    fragment_hamiltonians = {}
    for i in range(len(fragments)):
        fragment_hamiltonians[str(i)] = np.array(hamiltonian_data['i0f'+str(i)])
    return fragment_hamiltonians

def loss_function(potential_coeffs, fragment, penalty_factor):
    ground_state = get_perturbed_gs(potential_coeffs, fragment)
    H = fragment_hamiltonians[fragment]
    energy_expectation = np.conjugate(ground_state.transpose())@(H@ground_state) 
    
    ground_state = ground_state.flatten()
    
    penalty = 0
    for adjacent in fragment_overlap[fragment]:
        rho_edge = []
        rho_center = []
        for site in fragment_overlap[fragment][adjacent]:
            indices_to_trace = list(set(range(num_qubits)) - {site[0]})
            rho_edge.append(np.matrix(qi.partial_trace(qi.DensityMatrix(ground_state), indices_to_trace)))

            indices_to_trace = list(set(range(num_qubits)) - {site[1]})
#             print(fragment_gs[adjacent])
            rho_center.append(np.matrix(qi.partial_trace(qi.DensityMatrix(fragment_gs[adjacent]), indices_to_trace)))
            
            penalty += np.trace((rho_edge[-1]-rho_center[-1]).H@(rho_edge[-1]-rho_center[-1]))
    penalty *= penalty_factor
    
    total_loss = np.real((energy_expectation + penalty)[0,0])
#     print(total_loss)
    return total_loss

def get_perturbed_gs(potential_coeffs, fragment):
    '''
    SLIGHTLY HARD-CODED
    '''
    if fragment == '0':
        potential_term = construct_update_matrix(np.array([[0,0],[0,1]]), (2,0)) + construct_update_matrix(np.array([[0,0],[0,1]]), (8,0))
        H = fragment_hamiltonians[fragment] + potential_coeffs[0]*potential_term
    elif fragment == '1':
        potential_term = construct_update_matrix(np.array([[0,0],[0,1]]), (0,0)) + construct_update_matrix(np.array([[0,0],[0,1]]), (6,0))
        H = fragment_hamiltonians[fragment] + potential_coeffs[1]*potential_term
        potential_term = construct_update_matrix(np.array([[0,0],[0,1]]), (2,2)) + construct_update_matrix(np.array([[0,0],[0,1]]), (8,0))
        H += potential_coeffs[2]*potential_term
    elif fragment == '2':
        potential_term = construct_update_matrix(np.array([[0,0],[0,1]]), (0,0)) + construct_update_matrix(np.array([[0,0],[0,1]]), (6,0))
        H = fragment_hamiltonians[fragment] + potential_coeffs[3]*potential_term
        potential_term = construct_update_matrix(np.array([[0,0],[0,1]]), (2,2)) + construct_update_matrix(np.array([[0,0],[0,1]]), (8,0))
        H += potential_coeffs[4]*potential_term
    elif fragment == '3':
        potential_term = construct_update_matrix(np.array([[0,0],[0,1]]), (0,0)) + construct_update_matrix(np.array([[0,0],[0,1]]), (6,0))
        H = fragment_hamiltonians[fragment] + potential_coeffs[5]*potential_term
        potential_term = construct_update_matrix(np.array([[0,0],[0,1]]), (2,2)) + construct_update_matrix(np.array([[0,0],[0,1]]), (8,0))
        H += potential_coeffs[6]*potential_term
    elif fragment == '4':
        potential_term = construct_update_matrix(np.array([[0,0],[0,1]]), (0,0)) + construct_update_matrix(np.array([[0,0],[0,1]]), (6,0))
        H = fragment_hamiltonians[fragment] + potential_coeffs[7]*potential_term
        potential_term = construct_update_matrix(np.array([[0,0],[0,1]]), (2,2)) + construct_update_matrix(np.array([[0,0],[0,1]]), (8,0))
        H += potential_coeffs[8]*potential_term
    elif fragment == '5':
        potential_term = construct_update_matrix(np.array([[0,0],[0,1]]), (0,0)) + construct_update_matrix(np.array([[0,0],[0,1]]), (6,0))
        H = fragment_hamiltonians[fragment] + potential_coeffs[9]*potential_term


    
    H = scipy.sparse.csr_matrix(H)
    eigvals, eigvec = scipy.sparse.linalg.eigsh(H, which = 'SA')
    min_eigval_index = np.argmin(eigvals)
    return eigvec[:,min_eigval_index].reshape((-1,1))
    
def compute_gs_energy(ground_state,H):
    ground_state = ground_state.reshape((-1,1))
    return np.real(np.conjugate(ground_state.transpose())@(H@ground_state))[0,0]

def compute_rmse():
    rmse = 0
    num_sites = 0
    for f in fragments:
#         print('frag', f, len(fragment_gs[f]))
        for adjacent in fragment_overlap[f]:
#             print('adj', adjacent)
            rho_edge = []
            rho_center = []
            for site in fragment_overlap[f][adjacent]:
#                 print('site',site)
                indices_to_trace = list(set(range(num_qubits)) - {site[0]})
                rho_edge.append(np.matrix(qi.partial_trace(qi.DensityMatrix(fragment_gs[f]), indices_to_trace)))

                indices_to_trace = list(set(range(num_qubits)) - {site[1]})
                rho_center.append(np.matrix(qi.partial_trace(qi.DensityMatrix(fragment_gs[adjacent]), indices_to_trace)))

                rmse += np.trace((rho_edge[-1]-rho_center[-1]).H@(rho_edge[-1]-rho_center[-1]))
                num_sites += 1
    rmse /= num_sites
    rmse = np.sqrt(rmse)
    return np.real(rmse)

def construct_update_matrix(rho, site, num_qubits = num_qubits):
    '''
     construct the matrix which will be added to the existing fragment hamiltonian during a gradient descent iteration
     This matrix is equal to (rho_CB-rho_EA) tensor product identity 
    '''
    num_fragment_qubits = num_qubits
    edge_qubit = site[0]
    rho_size = 1 # number of qubits in rho

    qubit = num_fragment_qubits-1
    if edge_qubit == qubit:
        update_matrix = rho
        qubit -= rho_size
    else:
        update_matrix = np.eye(2)
        qubit -= 1

    while qubit >= 0:
        if qubit == edge_qubit:
            update_matrix = np.kron(update_matrix, rho)
            qubit -= rho_size
        else:
            update_matrix = np.kron(update_matrix, np.eye(2))
            qubit -= 1
    return update_matrix

def update_fragment_hamiltonian(H, overlap_indices, rho_edge, rho_center, delta_lambda, lr):
    '''
    after computing the gradient, complete the gradient descent step by updating the fragment hamiltonian with a perturbation
    '''
    new_H = H.copy()
    for i in range(len(rho_center)):
#         new_update = lr  * delta_lambda * construct_update_matrix(rho_center[i]-rho_edge[i], overlap_indices[i])
#         new_update = -lr  * (-delta_lambda)**5 * construct_update_matrix(rho_center[i]-rho_edge[i], overlap_indices[i])
        new_update = -lr * construct_update_matrix(rho_center[i]-rho_edge[i], overlap_indices[i])
        #print(np.linalg.norm(new_update))
        new_H += new_update
    return new_H

def compute_total_occupation_number():
    occupation_number = 0
    for f in fragments:
        for site in center_sites[f]:
            indices_to_trace = list(set(range(num_qubits)) - {site})
            rho_CA = np.matrix(qi.partial_trace(qi.DensityMatrix(fragment_gs[f]), indices_to_trace))
            occupation_number += rho_CA[-1,-1]
    return np.real(occupation_number)
    
def compute_rescaled_energy():
    occupation_number = compute_total_occupation_number()
    scale_constant = target_occupation_number/occupation_number - 1
#     print(scale_constant)
    
    new_occupation_number = 0 
    energy = 0 
    for f in fragments:
        for site in center_sites[f]:
            indices_to_trace = list(set(range(num_qubits)) - {site})
            rho_CA = np.matrix(qi.partial_trace(qi.DensityMatrix(fragment_gs[f]), indices_to_trace))
            rho_CA += scale_constant*rho_CA[-1,-1]*np.matrix([[-1,0],[0,1]])
            new_occupation_number += rho_CA[-1,-1]
#             print(np.trace(rho_CA))
            H_CA = H_CAs[f][site]
            energy += np.trace(H_CA@rho_CA)
    print('occupation number:', new_occupation_number)
    return np.real(energy)
    
def compute_rescaled_rho_CAs():
    occupation_number = compute_total_occupation_number()
    scale_constant = target_occupation_number/occupation_number - 1
#     print(scale_constant)
    
    new_occupation_number = 0 
    rho_CAs = {}
    energy = 0 
    for f in fragments:
        rho_CAs[f] = {}
        for site in center_sites[f]:
            indices_to_trace = list(set(range(num_qubits)) - {site})
            rho_CA = np.matrix(qi.partial_trace(qi.DensityMatrix(fragment_gs[f]), indices_to_trace))
            rho_CA += scale_constant*rho_CA[-1,-1]*np.matrix([[-1,0],[0,1]])
            new_occupation_number += rho_CA[-1,-1]
            rho_CAs[f][site] = rho_CA
    print(new_occupation_number)
    return rho_CAs
###############################################################################
##INITIALIZE FRAGMENT HAMILTONIANS#############################################

# fragment_hamiltonians = init_fragment_hamiltonians()
fragment_hamiltonians = init_fragment_hamiltonians()
fragment_gs_energies = {} # energies of ground states of fragment hamiltonians
fragment_gs = {} # density matrix of ground state of fragment hamiltonians

fragment_gs_energies_starting, fragment_gs_starting = update_fragment_gs(fragment_hamiltonians)
# save the fragment_gs for all fragments, 12/13/2022
gs_filename = 'gs_vec_quad_run00.pkl'
gs_file = open(gs_filename, "wb")
pickle.dump(fragment_gs_starting, gs_file)
gs_file.close()

###############################################################################
def main_algorithm_quadratic(num_runs, penalty_initial, penalty_increase, thresh):
    rmse = [compute_rmse()]
    penalty_factors = [penalty_initial]
    run = 0 
    penalty_factor = penalty_initial
    energy_error = [np.abs(compute_rescaled_energy()-target_energy)]
    potential_coeffs = np.zeros(10)
    
    while rmse[-1] > thresh and run <= num_runs:
        print('run number:', run, 'rmse:', rmse[-1],'energy error', energy_error[-1], 'penalty factor', penalty_factor)
        for f in fragments:
            print('fragment:', f)
            res = scipy.optimize.minimize(loss_function, potential_coeffs, (f, penalty_factor), method = 'L-BFGS-B', tol = thresh,
                                          options = {'eps': thresh})
            if not res.success:
                print('did not converge')
                return rmse, penalty_factors
            potential_coeffs = res.x
            fragment_gs[f] = get_perturbed_gs(res.x, f).flatten()
            #print(fragment_gs[f], np.linalg.norm(fragment_gs[f]))
            fragment_gs[f] /= np.linalg.norm(fragment_gs[f])
            fragment_gs_energies[f] = compute_gs_energy(fragment_gs[f],fragment_hamiltonians[f])
           
        # save the fragment_gs for all fragments, 12/11/2022
        gs_filename = 'gs_vec_quad_run' + str(run) + '.pkl'
        gs_file = open(gs_filename, "wb")
        pickle.dump(fragment_gs, gs_file)
        gs_file.close()
 
        rmse.append(compute_rmse())
#         energy_error.append(compute_rescaled_rho_CAs())
        energy_error.append(np.abs(compute_rescaled_energy()-target_energy))
        #print('occupation number', compute_total_occupation_number(), 'energy error', energy_error[-1])
        penalty_factors.append(penalty_factor)
        penalty_factor *= penalty_increase
        run += 1
    return np.array(rmse), np.array(energy_error), penalty_factors

def diverges(avg_gradients, n):
    '''
    algorithm diverges if the last n gradients are increasing
    '''
    last_n_gradients = (avg_gradients[:,1])[-n:]
    
    if np.all(np.diff(last_n_gradients) > 0):
        return True
    else:
        return False
    
def main_algorithm_linear(max_runs, num_iter, lr_initial, lr_exponent, thresh, quit_if_diverge = False, verbose = True):
    run = 0
    
    diverge_n = 3
    num_eigensolver_calls = 0
    
    rmse = [compute_rmse()]
    energy_error = [np.abs(compute_rescaled_energy()-target_energy)]
    # find initial gradients of the lagrangian wrt the lagrange multipliers
    gradients = {} # example: {'01': 0.001} means that the gradient of the lagrangian wrt 
                    # the lagrange multiplier corresponding to the overlap of fragments 0 and 1 is 0.001
    for f in fragments:
        for adjacent in fragment_overlap[f]:
            delta_lambda = 0
            rho_edge = []
            rho_center = []
            for site in fragment_overlap[f][adjacent]:
                indices_to_trace = list(set(range(num_qubits)) - {site[0]})
                rho_edge.append(np.matrix(qi.partial_trace(qi.DensityMatrix(fragment_gs[f]), indices_to_trace)))


                indices_to_trace = list(set(range(num_qubits)) - {site[1]})
                rho_center.append(np.matrix(qi.partial_trace(qi.DensityMatrix(fragment_gs[adjacent]), indices_to_trace)))
                #print(f,adjacent,rho_edge,rho_center)
                delta_lambda += 2*np.trace(rho_edge[-1]@rho_center[-1]) - np.trace(rho_edge[-1]@rho_edge[-1]) - np.trace(rho_center[-1]@rho_center[-1])
            gradients[f+adjacent] = np.sqrt(delta_lambda)
        
    avg_gradients = [] # average of the various gradients of the lagrangian wrt the different lagrange multipliers
    avg_gradients.append([num_eigensolver_calls, np.mean(np.abs(np.array(list(gradients.values()))))])
    
#     eig_gap = [fragment_es['0']-fragment_gs_energies['0']]
    
    while  rmse[-1] > thresh and run < max_runs: # main loop through all the fragments, matching each fragment to those overlapping with it
        if verbose: print('run number:', run, 'rmse:', rmse[-1], 'energy error', energy_error[-1])#lr_initial*((np.power(10,1/lr_exponent))**run))
        gradients = {} 
        for f in fragments: # loop through all the fragments
#             print('fragment:', f)
            for n in range(num_iter): # complete num_iter iterations of gradient descent when matching the fragment to its neighbor
                lr = lr_initial*(1-n/num_iter)#lr_initial*((np.power(10,1/lr_exponent))**run) * (1-n/num_iter) # reduce the learning rate at each step
#                 lr = lr_initial*((np.power(10,1/lr_exponent))**run) * (1-n/num_iter)
#                 lr = (lr_initial+lr_exponent*run) * (1-n/num_iter)
            #                 print('GD iteration:', n, lr)
                for adjacent in fragment_overlap[f]: # loop through all the neighboring fragments which overlap with the fragment of interest
#                     print('neighbor', adjacent)
                    delta_lambda = 0 # gradient of the lagrangian with respect to this lagrange multiplier
                    rho_edge = [] # list of all the 2-by-2 single-qubit density matrices of the "edge" sites on the fragment of interest
                    rho_center = [] # list of all the 2-by-2 single-qubit density matrices of the "center" sites on the neighboring fragment
                    for site in fragment_overlap[f][adjacent]: # loop through each single-qubit site of overlap between the two fragments
                        indices_to_trace = list(set(range(num_qubits)) - {site[0]})
                        rho_edge.append(np.matrix(qi.partial_trace(qi.DensityMatrix(fragment_gs[f]), indices_to_trace)))


                        indices_to_trace = list(set(range(num_qubits)) - {site[1]})
                        rho_center.append(np.matrix(qi.partial_trace(qi.DensityMatrix(fragment_gs[adjacent]), indices_to_trace)))
                        
                        # compute the gradient
                        delta_lambda += 2*np.trace(rho_edge[-1]@rho_center[-1]) - np.trace(rho_edge[-1]@rho_edge[-1]) - np.trace(rho_center[-1]@rho_center[-1])
                    
#                     print('delta_lambda', delta_lambda)
#                     print('rho edge', rho_edge)
#                     print('rho center', rho_center)
                    # update the Hamiltonian--this is the gradient descent update
                    new_H = update_fragment_hamiltonian(fragment_hamiltonians[f], fragment_overlap[f][adjacent], rho_edge, rho_center, 
                                                        delta_lambda, lr)
                    fragment_hamiltonians[f] = new_H
                    
#                     lagrange_multipliers[f][adjacent] += delta_lambda*lr
                    # recompute the new ground state of the updated fragment Hamiltonian
#                     gs_energy, gs = get_ground_state(fragment_hamiltonians[f])
#                     num_eigensolver_calls += 1
#                     fragment_gs_energies[f] = gs_energy
#                     fragment_gs[f] = qi.DensityMatrix(qi.Statevector(gs))
                    
#                     eigvals, eigvecs = np.linalg.eigh(fragment_hamiltonians[f])
#                     fragment_es[f] = eigvals[1]
                    
                    # record the gradient after the final step of gradient descent
                    if n == num_iter-1:
                        gradients[f+adjacent] = np.sqrt(delta_lambda)
        for f in fragments:
            gs_energy, gs = get_ground_state(fragment_hamiltonians[f])
            num_eigensolver_calls += 1
            fragment_gs_energies[f] = gs_energy
            fragment_gs[f] = gs
#             print('fragment num', f, len(gs))
          
        # save the fragment_gs for all fragments, 12/11/2022
        gs_filename = 'gs_vec_lin_run' + str(run) + '.pkl'
        gs_file = open(gs_filename, "wb")
        pickle.dump(fragment_gs, gs_file)
        gs_file.close()
            
        avg_gradients.append([num_eigensolver_calls, np.mean(np.abs(np.array(list(gradients.values()))))]) # compute the average of the gradients for a given run (through the main loop)
        rmse.append(compute_rmse())
        energy_error.append(np.abs(compute_rescaled_energy()-target_energy))
       # print('occupation number', compute_total_occupation_number(), 'energy error', energy_error[-1])
       # eig_gap.append(fragment_es['0']-fragment_gs_energies['0'])
        
        if quit_if_diverge and diverges(np.array(avg_gradients), diverge_n):
            if verbose: print('diverged')
            break
        run += 1

    return run, np.array(rmse), np.array(energy_error), (rmse[-1] <= thresh)#, np.array(eig_gap)


#############################################################
## MAIN #####################################################
#############################################################



############## HERE IS WHERE THE ACTUAL ALGORITHM IS RUN (QUAD PENALTY) ###########
num_runs = 0
penalty_initial = 1
penalty_increase = 25


thresh = 1e-8

fragment_hamiltonians = init_fragment_hamiltonians()
fragment_gs_energies = copy.deepcopy(fragment_gs_energies_starting)
fragment_gs = copy.deepcopy(fragment_gs_starting)

rmse, energy_error, penalty_factors = main_algorithm_quadratic(num_runs, penalty_initial, penalty_increase, thresh)

# np.save('rmse_H8.npy', rmse)
# np.save('energy_error_H8.npy', energy_error)
# np.save('penalty_factors_H8.npy', penalty_factors)
