"""
Methods for working with fragments and their Hamiltonians
"""
# imports
import copy

import numpy as np
import scipy
import time, itertools, h5py
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
from itertools import product
from pyscf import ao2mo
from qiskit.quantum_info import partial_trace, DensityMatrix, Statevector, Pauli

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

# local imports
from .fermionic_hamiltonian import fermionic_1rdm, frag2f1rdm

DATA_DIR = 'data/'


# read in the initial Hamiltonian matrix
def read_frag_ham(nao=6, num_frag=6, max_neighbors=2,
                  data_file=DATA_DIR+'h8_initial_hams_iter2.h5',
                  data_file_1pdm=DATA_DIR+'h8_initial_1pdm_iter2.h5',
                  FLAG_qubit_ham=False, FLAG_make_f1rdms=False):
    """
    f1rdms: (Global)
    nborfrag_list:
    :param nao:
    :param num_frag:
    :param data_file:
    :param FLAG_qubit_ham:
    :return:

    TODO: ham_qubit_mat_init actually contains the qubit Hamiltonians -- needs to be changed to fermionic Hams
    """
    # define some data structs
    f1rdms = np.zeros((num_frag, 2, nao, nao))  # all fermionic 1rdms
    nborfrag_list = np.zeros((num_frag, max_neighbors), dtype=np.int8)  # index for neighbor fragment centers
    num_nb = np.zeros(num_frag, dtype=np.int8)  # number of neighboring fragments

    if FLAG_qubit_ham:
        ham_qubit_mat_init = []  # initial matrix form of the qubit operators for each fragment hamiltonian

    for frag in range(num_frag):
        index1e = 'i0f'+str(frag)
        print('Fragment ' + str(frag))
        r = h5py.File(data_file, 'r')
        htot = np.array(r.get(index1e))
        r.close()

        if FLAG_qubit_ham:
            ham_qubit_mat_init.append(htot)

        if frag == 0:
            num_nb[frag] = 1
            nborfrag_list[frag][0] = np.array([frag+1])
        elif frag == num_frag - 1:
            num_nb[frag] = 1
            nborfrag_list[frag][0] = np.array([frag-1])
        else:
            num_nb[frag] = 2
            nborfrag_list[frag] = np.array([frag-1,frag+1])

        if FLAG_make_f1rdms:
            # precompute all fermionic 1rdms
            f1rdms[frag] = frag2f1rdm(ham_qubit_mat_init, frag, nao)
            file_1pdm.create_dataset(index1e, data = f1rdms[frag])
        else:
            # read in 1pdm from external file
            file_1pdm = h5py.File(data_file_1pdm, 'r')
            pdm1 = np.array(file_1pdm.get(index1e))
            f1rdms[frag] = pdm1
            #print(f1rdms[frag])
            print("trace of f1rdm = %12.8f, %12.8f" % (np.trace(f1rdms[frag][0]), np.trace(f1rdms[frag][1])))

    if FLAG_qubit_ham:
         return f1rdms, ham_qubit_mat_init, num_nb, nborfrag_list
    else:
        return f1rdms, num_nb, nborfrag_list


def calc_overlap(f1rdma, f1rdmb, ie=2, ic=1):
    """
    Fermionic -- Compute the overlap of the RDM edge sites of A (ie) with center sites of B (ic),
    Default, assume A is to the left of B, ie = 2, ic = 1

    Inputs:
        f1rdma:
        f1rdmb:
        ie:
        ic:

    Returns:
    """
    # Is this necessarily always 2?
    tempovlp = np.zeros(2)
    for ispin in range(2):
        # loop over spin for overlap
        tempovlp[ispin] += (f1rdma[ispin][ie, ie] - f1rdmb[ispin][ic, ic])

    return tempovlp


class qbe_fragment(object):
    """
    Defines the class qbe_fragment which describes all the fragments involved in the simulation
    """
    def __init__(self, n_frag, nao, max_neighbors,
                 data_file=DATA_DIR+'h8_initial_hams_iter2.h5',
                 data_file_1pdm=DATA_DIR+'h8_initial_1pdm_iter2.h5',
                 FLAG_qubit_ham=False, FLAG_make_f1rdms=False):
        """
        Assuming that this is initialized through a data file containing Fermionic hamiltonians (obtained through Schmidt decomposition)

        Assuming nao (number of atomic orbitals is same for all) for the moment

        Inputs:
            n_frag: number of fragments
            nao: number of atomic orbitals

        Notation:
            nb -- neighbors

        TODO: Generalize for unequal fragment sizes such as nao
        TODO: Works only with FLAG_qubit_ham=True ... Generalize
        """
        f1rdms, ham_qubit_mat_init, num_nb, nborfrag_list = read_frag_ham(nao=nao, num_frag=n_frag,
                                                                          max_neighbors=max_neighbors,
                                                                          data_file=data_file,
                                                                          data_file_1pdm=data_file_1pdm,
                                                                          FLAG_qubit_ham=FLAG_qubit_ham,
                                                                          FLAG_make_f1rdms=FLAG_make_f1rdms)
        self.n_frag = n_frag
        self.nao = nao
        self.max_neighbors = max_neighbors

        self.f1rdms = f1rdms
        self.qubit_hamiltonians = ham_qubit_mat_init

        # Number of neighbors for each fragment
        self.num_nb = num_nb

        # Contains indices of neighbors for each fragment
        self.frag_nb_list = nborfrag_list

    def frag2f1rdm(self, ind_frag, FLAG_verbose=True):
        # given fragment number, output the fermionic 1rdm of it
        eigval, eigvec = eigs(self.qubit_hamiltonians[ind_frag], k=1, which='SR')

        # sort the eigenvalues and the corresponding eigenvectors in descending order, last element is the ground state
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]

        if FLAG_verbose:
            print('E(ED in qubit basis, frag %d) = %.8f' % (ind_frag, eigval[-1].real))

        # construct the total density matrix from state vector
        fulldm = DensityMatrix(eigvec[:,-1])
        ferm1rdm = fermionic_1rdm(fulldm, self.nao)

        return ferm1rdm

    def calc_grad(self, ind_frag, icenter=1):
        # Fermionic version -- Compute the gradient matrix for the current fragment by looping over all its adjacent fragments
        # ferm1rdmA = frag2f1rdm(frag)

        ferm1rdmA = self.f1rdms[ind_frag]
        grad = np.zeros((2 * self.nao, 2 * self.nao))

        # Loop over all fragments that overlap with the current fragment, and compute the overlap to get the gradient
        for ind_nb in range(self.num_nb[ind_frag]):
            ferm1rdmB = self.f1rdms[self.frag_nb_list[ind_frag][ind_nb]]

            # get the gradient
            ovlpsiteA = icenter + (self.frag_nb_list[ind_frag][ind_nb] - ind_frag)
            tempovlp = calc_overlap(ferm1rdmA, ferm1rdmB, ovlpsiteA, icenter)

            for ispin in range(2):
                grad[ispin * self.nao + ovlpsiteA][ispin * self.nao + ovlpsiteA] = tempovlp[ispin]

        return grad


# Utility function that may be need to be sent elsewhere later
def get_ground_state(H):
    """
    returns ground state energy and statevector of a Hamiltonian matrix
    """
    if scipy.sparse.issparse(H):
        S, V = scipy.sparse.linalg.eigsh(H)

        ind = np.argmin(S)
        gs_energy = S[ind]
        gs_vec = V[:, ind]
    else:
        eigvals, eigvec = np.linalg.eigh(H)
        ind_min_eigval = np.argmin(eigvals)
        gs_energy = eigvals[ind_min_eigval]
        gs_vec = eigvec[:,ind_min_eigval]

    return gs_energy, gs_vec


class qbe_fragment_qubit(object):
    """
    Defines the class qbe_fragment which describes all the fragments involved in the simulation
    """
    def __init__(self, n_frag, n_qubits_frag, labels_fragments,
                 fragment_info, fragment_neighbors, fragment_H, target_fragment_H=None,
                 FLAG_track_generators=True):
        """
        Assuming that this is initialized through a dictionary of qubit hamiltonians

        Inputs:
            n_frag: number of fragments
            n_qubits_frag: number of qubits in each fragment (assuming to be the same for all!)
            fragment_info:
            target_fragment_H: dictionary of PauliSumOp representations of target Hamiltonians i.e., {'f': paulisumop}

            maximum number of neighbors of each fragment
                -- note the internal fragments will have more neighbors than the boundaries

        Notation:
            nb -- neighbors

        Assuming number of overlapping sites = 1 (qubit)

        TODO: Update code to account for more overlapping sites
        """
        self.n_frag = n_frag
        self.n_qubits = n_qubits_frag
        self.n_qubits_overlapping_site = 1
        self.labels = labels_fragments
        self.info = fragment_info
        self.H = fragment_H

        # Save the PauliSumOp of H
        if target_fragment_H is not None:
            self.paulisumop_H = copy.deepcopy(target_fragment_H)

            # convert to each PauliSumOp to a dictionary representation as well
            self.target_H = {f: {} for f in labels_fragments}
            for f in labels_fragments:
                _temp_ham = target_fragment_H[f].to_pauli_op()

                for ind in range(len(_temp_ham)):
                    # x is a PauliOp
                    x = _temp_ham[ind]
                    self.target_H[f].update({str(x.primitive): x.coeff})
        else:
            self.paulisumop_H = None
            self.target_H = None

        # Some error throwing
        if self.H[self.labels[0]].shape[0] != 2**n_qubits_frag:
            raise ValueError("Dimension mismatch between number of qubits in fragment and Hamiltonian being set!")

        # Information of neighbors for each fragment
        self.neighbors = fragment_neighbors

        # Create ground states and energies
        self.gs_energies = {f: 0.0 for f in labels_fragments}   # keys: fragment index, values: ground state energy
        self.ground_state = {f: np.zeros(shape=(2**n_qubits_frag, 2**n_qubits_frag), dtype=complex) for f in labels_fragments}

        self.update_ground_state(labels_fragments)

        # Get rho_edge (rho_E) and rho_center (rho_C) for all fragments
        # (assuming total number of overlapping sites is at max 2 with each neighbor)
        self.rho_E = {f: {} for f in self.labels}
        self.rho_C = {f: {} for f in self.labels}

        self.sigma_rho_E = {f: {} for f in self.labels}
        self.sigma_rho_C = {f: {} for f in self.labels}

        # Do we need to track generators of rho_E and rho_C at the overlapping sites?
        self.FLAG_track_generators = FLAG_track_generators
        if FLAG_track_generators:
            # Assuming overlap site is 1 qubit
            n_generators = 4**self.n_qubits_overlapping_site - 1
            pauli_strings = ['X', 'Y', 'Z']
            pauli_mat = [np.array(Pauli(P)) for P in pauli_strings]
            self.overlap_P = {'n': n_generators, 'pauli_strings': pauli_strings, 'pauli_mat': pauli_mat}

        for f in self.labels:
            for nb_f in self.neighbors[f]:
                self.rho_E[f].update({nb_f: [self.compute_partial_rho(f, site[0]) for site in self.neighbors[f][nb_f]['edge']]})
                self.rho_C[f].update({nb_f: [self.compute_partial_rho(f, site[0]) for site in self.neighbors[f][nb_f]['center']]})

                # The computation below at the moment assumes 1 overlapping site!
                if FLAG_track_generators:
                    sigma_rho_E_temp, sigma_rho_C_temp = self.pauli_expectations_overlapping_site(f, nb_f)
                    self.sigma_rho_E[f].update({nb_f: sigma_rho_E_temp})
                    self.sigma_rho_C[f].update({nb_f: sigma_rho_C_temp})

    def compute_partial_rho(self, f, site):
        """
        if site is edge site then this is rho_E, if site is center site then returns rho_C
        :param f:
        :param edge_site:
        :return:
        """
        indices_to_trace = list(set(range(self.n_qubits)) - {site})
        return np.matrix(partial_trace(self.ground_state[f], indices_to_trace))

    def pauli_expectations_overlapping_site(self, f, nb_f):
        """
        This code needs to be severly optimized!

        There will be multiple arrays for multiple sites
        """
        sigma_rho_E_temp = []
        sigma_rho_C_temp = []

        for ind_site in range(self.neighbors[f][nb_f]['n_sites']):
            # TODO: Check why you put real and if we reproduce everything correctly with chucking out real.
            # There shouldn't be any imaginary part in the gradient.
            sigma_rho_E_temp.append(np.array([
                np.real(np.trace(self.overlap_P['pauli_mat'][k] @ self.rho_E[f][nb_f][ind_site]))
                for k in range(self.overlap_P['n'])]))

            sigma_rho_C_temp.append(np.array([
                np.real(np.trace(self.overlap_P['pauli_mat'][k] @ self.rho_C[f][nb_f][ind_site]))
                for k in range(self.overlap_P['n'])]))

        return sigma_rho_E_temp, sigma_rho_C_temp

    def update_ground_state(self, list_fragments):
        """
        updates the ground state energies and ground state density matrices

        TODO: Give functionality for this to be updated through PhaseEstimation, VQE, etc.
        """
        for f in list_fragments:
            gs_energy, gs = get_ground_state(self.H[f])
            self.gs_energies[f] = gs_energy
            self.ground_state[f] = DensityMatrix(Statevector(gs))

    def update_partial_rho_fragment(self, list_fragments):
        for f in list_fragments:
            # update rho_E and rho_C for f with all nb_f
            for nb_f in self.neighbors[f]:
                for ind_site in range(self.neighbors[f][nb_f]['n_sites']):
                    site_E = self.neighbors[f][nb_f]['edge'][ind_site]
                    site_C = self.neighbors[f][nb_f]['center'][ind_site]

                    self.rho_E[f][nb_f][ind_site] = self.compute_partial_rho(f, site_E[0])
                    self.rho_C[f][nb_f][ind_site] = self.compute_partial_rho(f, site_C[0])

                if self.FLAG_track_generators:
                    sigma_rho_E_temp, sigma_rho_C_temp = self.pauli_expectations_overlapping_site(f, nb_f)
                    self.sigma_rho_E[f][nb_f] = sigma_rho_E_temp
                    self.sigma_rho_C[f][nb_f] = sigma_rho_C_temp


def qubit_rdm(rdm_q, fulldm, nao, bath_orbital_list=[3,4,5,9,10,11]):
    # construct the reduced density matrix on each fragment site for spin up and down
    numsite = int(nao/2)
    tempdm = partial_trace(fulldm, bath_orbital_list)

    trindex = np.arange(nao)
    for ispin in range(2):
        for isite in range(numsite):
          trindex = trindex.pop(ispin*numsite + isite)
          print(trindex)

          # list in block-spin format, 6 up spin-orbitals, followed by 6 down spin-orbitals
          rdm_q[ispin*numsite + isite] = partial_trace(tempdm, trindex)

    return rdm_q