#!/usr/bin/env python
# coding: utf-8

# ## QBE with VMC solver from PyQMC and PySCF
# Yuan Liu
# 10/27/2022
# (adapted from the QBE script for linear matching of H4 by Z.C.)

# In[2]:



# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import h5py
import sys
import copy

import glob
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from pyscf import gto, scf, ao2mo
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyqmc.api as pyq
import pyqmc.obdm



# In[3]:


num_orbs = 6 # number of spatial orbitals in the Hamiltonian, the size of the embedding space (frag + entangled bath)
num_eles = 6 # number of electrons in the fragment Hamiltonian


# In[4]:


# keys: fragment labels
# values: list of indices of qubits in the fragment
# example: {'0': [0,1,2,6,7,8]} means that fragment '0' contains qubits with indices 0, 1, 2, 6, 7, and 8
fragments = {'0': [0,1,2,3,4,5],'1': [0,1,2,3,4,5], '2': [0,1,2,3,4,5],'3': [0,1,2,3,4,5],'4': [0,1,2,3,4,5],'5': [0,1,2,3,4,5]} 

# Let fragment A be the main fragment, then let B be any fragment which overlaps with A
# keys: label of fragment A
# values: dict, with
    # keys: label of fragment B
    # values: list of ordered qubit pairs on fragments A and B which overlap

# example: {'0': {'1': [[2,1],[8,7]]}} means that fragment '0' and fragment '1' overlap at two sites: qubit 2 on
    # fragment A overlaps with qubit 1 on fragment B; also, qubit 8 on fragment A overlaps with qubit 7 on fragment B.
fragment_overlap = {
                    '0': {'1': [[2,1]]}, 
                    '1': {'0': [[0,1]], '2': [[2,1]]},
                    '2': {'1': [[0,1]], '3': [[2,1]]},
                    '3': {'2': [[0,1]], '4': [[2,1]]},
                    '4': {'3': [[0,1]], '5': [[2,1]]},
                    '5': {'4': [[0,1]]}
                   }



# In[5]:


# # load the Hamiltonian matrices for each fragment
hamiltonian_data_h1 = h5py.File('../../data/H8_h1.h5', 'r')
hamiltonian_data_h2 = h5py.File('../../data/H8_eri_file.h5', 'r')


# In[6]:


def init_fragment_hamiltonians():
    '''
    initialize a dictionary where the keys are fragment labels 
    and the values are the corresponding fragment Hamiltonians for 1- and 2-e integrals
    done.
    '''
    fragment_hamiltonians_h1 = {}
    fragment_hamiltonians_h2 = {}
    for i in range(len(fragments)):
        fragment_hamiltonians_h1[str(i)] = np.array(hamiltonian_data_h1['i0f'+str(i)])
        fragment_hamiltonians_h2[str(i)] = np.array(hamiltonian_data_h2['f'+str(i)])
    return fragment_hamiltonians_h1, fragment_hamiltonians_h2


# In[7]:


# Utility functions for VMC with PyQMC
        

def mean_field(chkfile, h1, h2, nele, nao):
    '''
        Given checkfile name, 1- and 2-e integral, nele, nao, generate the mean field solution and store it
    '''
    
    mol = gto.Mole()
    #mol.verbose  = 5 # 5 for debug
    mol.atom = '''
      H 0.0 0.0 0.0
      H 0.0 0.0 1.0
      H 0.0 0.0 2.0
      H 0.0 0.0 3.0
      H 0.0 0.0 4.0
      H 0.0 0.0 5.0
    '''
    mol.nelectron = nele
    mol.basis    = 'sto-3g'
    mol.unit     = 'A'
    mol.charge   = 0
    mol.spin     = 0
    mol.symmetry = False
    mol.incore_anyway = True
    mol.build()
    
    # print(np.linalg.eig(h1))
    #mf = scf.RHF(mol).newton()
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(nao)
    # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    #symm = 8 # change to 8 for 8-fold symmetry
    #h2 = ao2mo.restore(symm, h2, nao)
    mf._eri = h2
    #mf._eri = np.zeros((2,2,2,2))

    #print(h1)
    #print(h2)
    mf.chkfile = chkfile
    mf.kernel()
    

def avg(vec):
    '''
        Computing the average and standard deviation of a vector
    '''
    nblock = vec.shape[0]
    avg = np.mean(vec,axis=0)
    std = np.std(vec,axis=0)
    return avg, std/np.sqrt(nblock)


def mo2lo_trans(mat, mf_file):
    '''
        Given mat as the 1RDM in MO basis, output mat_lo in LO basis by 
        transforming the 1RDM to LO representation using mo_coeff
    '''
    r = h5py.File(mf_file,'r')
    mo_coeff = np.array(r.get('scf/mo_coeff'))
    #print(mo_coeff)
    mat_lo = np.matmul(mo_coeff, np.matmul(mat, np.transpose(mo_coeff)))
    return mat_lo



    


# In[8]:


# def get_ground_state(H):
#     '''
#     returns ground state energy and statevector of a Hamiltonian matrix
#     '''
#     eigvals, eigvec = np.linalg.eigh(H)
#     ground_state_energy = eigvals[0]
#     ground_state = eigvec[:,0]
#     return ground_state_energy, ground_state


# def update_fragment_gs(fragment_hamiltonians): 
#     '''
#     updates the fragment ground state energies and fragment ground state density matrices
#     '''
#     fragment_gs_energies = {} # dictionary. keys: fragment labels, values: energy of fragment ground state
#     fragment_gs = {} # dictionary. keys: fragment labels, values: fragment ground state density matrices
#     for f in fragment_hamiltonians:
#         gs_energy, gs = get_ground_state(fragment_hamiltonians[f])
#         fragment_gs_energies[f] = gs_energy
#         fragment_gs[f] = qi.DensityMatrix(qi.Statevector(gs))
#     return fragment_gs_energies, fragment_gs

# Define VMC version of these functions

def get_ground_state_vmc(prefix, h1, h2, num_blocks = 30):
    '''
    returns ground state energy, 1-RDM (under computational basis), and the associated errors of a Hamiltonian matrix by accumulating num_sam
    Input: prefix is a string to indicate fragment information and BE iteration information
    '''
    
    
    mean_field(prefix+"mf.hdf5", h1, h2, num_eles, num_orbs)

    pyq.OPTIMIZE(prefix+"mf.hdf5", #Construct a Slater-Jastrow wave function from the pyscf output
                 prefix+"optimized_wf.hdf5", #Store optimized parameters in this file.
                 nconfig=100, #Optimize using this many Monte Carlo samples/configurations
                 max_iterations=10,   #10 optimization steps
                 verbose=False)

    #df = pyq.read_opt(prefix+"optimized_wf.hdf5")
    #print(df)

    with h5py.File(prefix+"optimized_wf.hdf5") as f:
        print("keys", list(f.keys()))
        print("wave function parameters", list(f['wf'].keys()))
        print(f['wf/wf2bcoeff'][()])


    pyq.VMC(prefix+"mf.hdf5",
            prefix+"vmc_data.hdf5",                      #Store Monte Carlo data in this file
            load_parameters=prefix+"optimized_wf.hdf5", #Load optimized parameters from this file
            nblocks=num_blocks,                        #Run for num_blocks blocks. By default, there are 10 steps per block and 1000 configurations i
            accumulators={'rdm1':True},
            verbose=True)


    #print(pyq.read_mc_output("vmc_data.hdf5"))
    
    with h5py.File(prefix+"vmc_data.hdf5") as f: 
        warmup=2
        en, en_err = avg(f['energytotal'][warmup:,...])
        rdm1up, rdm1up_err=avg(f['rdm1_upvalue'][warmup:,...])
        rdm1upnorm, rdm1upnorm_err = avg(f['rdm1_upnorm'][warmup:,...])
        rdm1up = pyqmc.obdm.normalize_obdm(rdm1up,rdm1upnorm)
        rdm1up_err=pyqmc.obdm.normalize_obdm(rdm1up_err,rdm1upnorm)

        rdm1down, rdm1down_err=avg(f['rdm1_downvalue'][warmup:,...])
        rdm1downnorm, rdm1downnorm_err = avg(f['rdm1_downnorm'][warmup:,...])
        rdm1down = pyqmc.obdm.normalize_obdm(rdm1down,rdm1downnorm)
        rdm1down_err=pyqmc.obdm.normalize_obdm(rdm1down_err,rdm1downnorm)

        ## transform to LO basis
        rdm1up_lo = mo2lo_trans(rdm1up, prefix+"mf.hdf5")
        rdm1up_lo_err = mo2lo_trans(rdm1up_err, prefix+"mf.hdf5")
        rdm1down_lo = mo2lo_trans(rdm1down, prefix+"mf.hdf5")
        rdm1down_lo_err = mo2lo_trans(rdm1down_err, prefix+"mf.hdf5")
        
    ground_state_1rdm = (rdm1up_lo + rdm1down_lo) * 0.5
    ground_state_1rdm_err = (rdm1up_lo_err + rdm1down_lo_err) * 0.5
    ground_state_energy = en
    ground_state_energy_err = en_err
    

    return ground_state_energy, ground_state_energy_err, ground_state_1rdm, ground_state_1rdm_err



def update_fragment_gs_vmc(fragment_hamiltonians_h1, fragment_hamiltonians_h2, num_samp): 
    '''
    updates the fragment ground state energies and fragment ground state density matrices
    done.
    '''
    fragment_gs_energies = {} # dictionary. keys: fragment labels, values: energy of fragment ground state
    fragment_gs_energies_err = {} # dictionary. keys: fragment labels, values: energy err of fragment ground state
    fragment_gs_1rdm = {} # dictionary. keys: fragment labels, values: fragment ground state density matrices
    fragment_gs_1rdm_err = {} # dictionary. keys: fragment labels, values: fragment ground state 1rdm error
    
    for f in fragment_hamiltonians_h1:
        print("VMC solver for fragment "+f)
        gs_energy, gs_energy_err, gs_1rdm, gs_1rdm_err = get_ground_state_vmc('frag'+str(f)+'_', 
                                                                              fragment_hamiltonians_h1[f], 
                                                                              fragment_hamiltonians_h2[f], 
                                                                              num_samp)
        fragment_gs_energies[f] = gs_energy
        fragment_gs_energies_err[f] = gs_energy_err
        fragment_gs_1rdm[f] = gs_1rdm
        fragment_gs_1rdm_err[f] = gs_1rdm_err
    return fragment_gs_energies, fragment_gs_energies_err, fragment_gs_1rdm, fragment_gs_1rdm_err


def get_num_samples_vmc(num_samp_init, n_iter, exp_base=None, power_index=None, flag_sample_schedule = 'const'):
    '''
    Given the initial sample and flags, return the number of samples to take in current iteration.
    '''
    
    if flag_sample_schedule == 'const':
        num_samp = num_samp_init
    elif flag_sample_schedule == 'power':
        num_samp = num_samp_init * n_iter**power_index
    elif flag_sample_schedule == 'exp':
        num_samp = num_sample_init * exp_base**n_iter
    else:
        print('Sampling schedule flag = ', flag_sample_schedule, ' not implemented.')
        exit()
        
    return num_samp
    
    
    
    


# In[9]:


# def update_fragment_hamiltonian(H, overlap_indices, rho_edge, rho_center, delta_lambda, lr):
#     '''
#     after computing the gradient, complete the gradient descent step by updating the fragment hamiltonian with a perturbation
#     '''
#     new_H = H.copy()
#     for i in range(len(rho_center)):
#         new_update = lr * 2 * delta_lambda * construct_update_matrix(rho_center[i]-rho_edge[i], overlap_indices[i])
#         #print(new_update)
#         new_H += new_update
#     return new_H


# now code the VMC version
def construct_update_matrix_vmc(rdm1_center, rdm1_edge, ovlp_idx):
    '''
     construct the matrix which will be added to the existing fragment hamiltonian during a gradient descent iteration
     This matrix is the same size as h1, and equal to (1rdm_CB-1rdm_EA) with the rest being zeros.
     done.
    '''
    num_fragment_orbs = num_orbs
    update_matrix = np.zeros((num_fragment_orbs,num_fragment_orbs))
    
    overlap_size = len(ovlp_idx)
    edge_list = []
    center_list = []
    for i in range(overlap_size):
        edge_list.append(ovlp_idx[i][0])
        center_list.append(ovlp_idx[i][1])
        
    for i in range(overlap_size):
        for j in range(i, overlap_size):
            update_matrix[edge_list[i],edge_list[j]] = rdm1_center[center_list[i], center_list[j]] \
            - rdm1_edge[edge_list[i], edge_list[j]]
            update_matrix[edge_list[j],edge_list[i]] = np.conj(update_matrix[edge_list[i],edge_list[j]])

    
    return update_matrix

def update_fragment_hamiltonian_vmc(h1, overlap_indices, rdm1_edge, rdm1_center, lr):
    '''
    after computing the gradient, complete the gradient descent step by updating 
    the one-body part of the fragment hamiltonian with a perturbation.
    done.
    '''
    new_h1 = h1.copy()
    new_update = lr * construct_update_matrix_vmc(rdm1_center, rdm1_edge, overlap_indices)
    #print(new_update)
    new_h1 += new_update
    return new_h1



# In[10]:


def diverges(avg_gradients, n):
    '''
    algorithm diverges if the last n gradients are increasing
    '''
    last_n_gradients = (avg_gradients[:,1])[-n:]
    
    if np.all(np.diff(last_n_gradients) > 0):
        return True
    else:
        return False
    
def main_algorithm(max_runs, num_iter, lr_initial, thresh, 
                   quit_if_diverge = False, verbose = False):
    run = 0
    num_samp_run = get_num_samples_vmc(num_samp_init, run, exp_base=None, power_index=None, flag_sample_schedule = 'const')
    
    diverge_n = 3
    num_eigensolver_calls = 0
    
    # find initial gradients of the lagrangian wrt the lagrange multipliers
    gradients = {} # example: {'01': 0.001} means that the gradient of the lagrangian wrt 
                    # the lagrange multiplier corresponding to the overlap of fragments 0 and 1 is 0.001
    for f in fragments:
        for adjacent in fragment_overlap[f]:
            delta_lambda = construct_update_matrix_vmc(fragment_gs_1rdm[adjacent], fragment_gs_1rdm[f],
                                                       fragment_overlap[f][adjacent])
            gradients[f+adjacent] = delta_lambda
        
    avg_gradients = [] # average of the various gradients of the lagrangian wrt the different 
                       # lagrange multipliers
    avg_gradients.append([num_eigensolver_calls, np.mean(np.abs(np.array(list(gradients.values()))))])
    
    acc_energies = []
    acc_1rdm = []
    
    acc_energies.append([num_eigensolver_calls, np.array(list(fragment_gs_energies.values())), np.array(list(fragment_gs_energies_err.values()))])
    acc_1rdm.append([num_eigensolver_calls, np.array(list(fragment_gs_1rdm.values())), np.array(list(fragment_gs_1rdm_err.values()))])
    #acc_energies.append([num_eigensolver_calls, list(fragment_gs_energies.values()), list(fragment_gs_energies_err.values())])
    #acc_1rdm.append([num_eigensolver_calls, list(fragment_gs_1rdm.values()), list(fragment_gs_1rdm_err.values())])

    while  avg_gradients[-1][1] > thresh and run < max_runs: # main loop through all the fragments, 
        #matching each fragment to those overlapping with it
        #if verbose: print('run number:', run, 'lr initial:', lr_initial*((np.power(10,1/lr_exponent))**run))
        # lr_initial = lr_initial / (run+1.0)**1
        if run == 30:
            lr_initial = lr_initial / 2.0
        elif run == 100:
            lr_initial = lr_initial / 2.0
        if verbose: print('\n run number:', run, 'lr initial:', lr_initial)
        gradients = {} 
        for f in fragments: # loop through all the fragments
            #print('fragment:', f)
            for n in range(num_iter): # complete num_iter iterations of gradient descent when matching the fragment to its neighbor
                #lr = lr_initial*((np.power(10,1/lr_exponent))**run) * (1-n/num_iter) # reduce the learning rate at each step
                lr = lr_initial * (1-n/num_iter)
                #print('GD iteration:', n, lr)
                for adjacent in fragment_overlap[f]: # loop through all the neighboring fragments which overlap with the fragment of interest
                    #print('neighbor', adjacent)
                    
                    # update the Hamiltonian--this is the gradient descent update
                    new_h1 = update_fragment_hamiltonian_vmc(fragment_hamiltonians_h1[f], fragment_overlap[f][adjacent], 
                                                            fragment_gs_1rdm[f], fragment_gs_1rdm[adjacent], lr)
                    fragment_hamiltonians_h1[f] = new_h1
                    
                    #lagrange_multipliers[f][adjacent] += delta_lambda*lr
                    # recompute the new ground state of the updated fragment Hamiltonian
                    prefix_name = 'frag'+str(f)+'_run'+str(run)+'_nb'+str(adjacent)+'_'
                    gs_energy, gs_energy_err, gs_1rdm, gs_1rdm_err = get_ground_state_vmc(prefix_name,
                                                                    fragment_hamiltonians_h1[f], 
                                                                    fragment_hamiltonians_h2[f], 
                                                                    num_samp_run)
                    num_eigensolver_calls += 1
                    fragment_gs_energies[f] = gs_energy
                    fragment_gs_energies_err[f] = gs_energy_err
                    fragment_gs_1rdm[f] = gs_1rdm
                    fragment_gs_1rdm_err[f] = gs_1rdm_err
                    
                    # record the gradient after the final step of gradient descent
                    if n == num_iter-1:
                        gradients[f+adjacent] = construct_update_matrix_vmc(fragment_gs_1rdm[adjacent], fragment_gs_1rdm[f],
                                                       fragment_overlap[f][adjacent])
        print('average gradient: ', np.mean(np.abs(np.array(list(gradients.values())))))
        print('acc energies: ', np.mean(np.array(list(fragment_gs_energies.values()))), np.linalg.norm(np.array(list(fragment_gs_energies_err.values()))))
        avg_gradients.append([num_eigensolver_calls, np.mean(np.abs(np.array(list(gradients.values()))))]) # compute the average of the gradients for a given run (through the main loop)
        acc_energies.append([num_eigensolver_calls, np.array(list(fragment_gs_energies.values())), np.array(list(fragment_gs_energies_err.values()))])
        acc_1rdm.append([num_eigensolver_calls, np.array(list(fragment_gs_1rdm.values())), np.array(list(fragment_gs_1rdm_err.values()))])
        
        if quit_if_diverge and diverges(np.array(avg_gradients), diverge_n):
            if verbose: print('diverged')
            break
        run += 1

        
        
    return run, np.array(avg_gradients), (avg_gradients[-1][1] <= thresh), np.array(acc_energies), np.array(acc_1rdm)


# In[ ]:





# In[11]:


# fragment_hamiltonians = init_fragment_hamiltonians()
fragment_hamiltonians_h1, fragment_hamiltonians_h2 = init_fragment_hamiltonians()
fragment_gs_energies = {} # energies of ground states of fragment hamiltonians
fragment_gs_energies_err = {}
fragment_gs_1rdm = {} # 1-rdm of ground state of fragment hamiltonians
fragment_gs_1rdm_err = {}
num_sample_list = {}

num_samp_init = 64

fragment_gs_energies_starting, fragment_gs_energies_err_starting, \
fragment_gs_1rdm_starting, fragment_gs_1rdm_err_starting \
= update_fragment_gs_vmc(fragment_hamiltonians_h1, fragment_hamiltonians_h2, num_samp_init)


# In[12]:


# Here we reset the files
for fname in glob.glob('*mf.hdf5') + glob.glob('*optimized_wf.hdf5') + glob.glob('*vmc_data.hdf5') + glob.glob('*dmc.hdf5'):
    if os.path.isfile(fname):
        os.remove(fname)


        
num_iter = 1 # number of gradient descent iterations
max_runs = 100 # number of runs, i.e. number of times the algorithm loops through all the fragments
lr_initial = 0.05 # initial gradient descent learning rate


thresh = 1e-16 # threshold for convergence, algorithm stops when the average gradient of the lagrangian wrt the lagrange multipliers falls below this value



# In[13]:


fragment_hamiltonians_h1, fragment_hamiltonians_h2  = init_fragment_hamiltonians()
fragment_gs_energies = copy.deepcopy(fragment_gs_energies_starting)
fragment_gs_energies_err = copy.deepcopy(fragment_gs_energies_err_starting)
fragment_gs_1rdm = copy.deepcopy(fragment_gs_1rdm_starting)
fragment_gs_1rdm_err = copy.deepcopy(fragment_gs_1rdm_err_starting)


# In[14]:


run, avg_gradients, complete, acc_energies, acc_1rdm = main_algorithm(max_runs, num_iter, lr_initial, 
                                              thresh, quit_if_diverge = False, verbose=True)


# In[15]:


if complete:
    print('complete, converged to', thresh)
else:
    print('incomplete, did not converge')


# In[16]:


# save data to txt
np.savetxt('avg_gradients.txt', np.array(avg_gradients))
np.savetxt('acc_energies.txt', np.array(acc_energies), fmt='%s')
np.savetxt('acc_1rdm.txt', np.array(acc_1rdm), fmt='%s')


# In[17]:


plot_data = np.transpose(np.array(avg_gradients))
#conv_limit = np.power(1/np.sqrt(np.power(10,1/lr_exponent)),np.arange(len(plot_data[1,:])))
plt.figure(dpi = 300)
#plt.semilogy(plot_data[0,:],(plot_data[1,:]),label='Perturbed H4')
plt.semilogy(np.arange(len(plot_data[1,:])), plot_data[1,:],label='H8')
#plt.semilogy(plot_data[0,:],conv_limit**2, label = '${(1/\sqrt{\gamma})}^{2\cdot\mathrm{iter}}$')
#plt.xlabel('number of calls to the eigensolver subroutine')
plt.xlabel('Number of Iterations')
plt.ylabel('Gradient \n(average over fragments)')
plt.title('Bootstrap Embedding with VMC solver for H8')
plt.grid()
plt.legend()
plt.savefig('H8_convergence_rate_VMC_run100_difflr_BEiter_Samp64_lr0d05.png')


