# imports
import copy
import sys
import numpy as np
import time, itertools, h5py
from numpy import linalg
from scipy.sparse.linalg import eigs
from itertools import product
from pyscf import ao2mo
import time

from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
from qiskit.quantum_info import partial_trace, DensityMatrix

from qiskit.opflow import (PauliOp, SummedOp, PauliExpectation, PauliSumOp)

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

# local imports
from . import lagrange_optimization as la_opt
from . import fragment_hamiltonian as frag_ham
from .fragment_hamiltonian import qbe_fragment, qbe_fragment_qubit

DATA_DIR = '../data/'


# Updates to the Hamiltonian
def lmbd2ham(lmbd, nao):
    # inumpyut the lagragian multiplier, and output the qubit hamiltonian corresponding to the local potentials

    electronic_energy_from_ints = ElectronicEnergy.from_raw_integrals(
        ElectronicBasis.SO, lmbd, np.zeros((2 * nao, 2 * nao, 2 * nao, 2 * nao))
    )
    # print(electronic_energy_from_ints)
    # print(lmbd)

    ferOp = electronic_energy_from_ints.second_q_ops()[0]  # here, output length is always 1
    qubit_converter = QubitConverter(mapper=JordanWignerMapper())
    qubitOp = qubit_converter.convert(ferOp)  # this will not give two-qubit reduction
    # print(qubitOp)
    # print(qubitOp.num_qubits)
    ham = qubitOp.to_matrix()
    # print(ham[])
    return ham


# def update_ham_f1rdm(mylambda, nao, num_frag, ham_qubit_mat_init):
#     for frag in range(num_frag):
#         ham_qubit_mat[frag] = ham_qubit_mat_init[frag] + lmbd2ham(mylambda[frag])
#         f1rdms[frag] = frag2f1rdm(frag, nao, ham_qubit_mat)
#
#     return ham_qubit_mat, f1rdms


def tot_err(fragment: qbe_fragment):
    # compute the total RMS error

    # Counts the total number of unique pairs of fragments
    n_const = 2 * (fragment.n_frag - 2) + 2

    error = 0.0
    for ind_frag in range(fragment.n_frag):
        error += (linalg.norm(fragment.calc_grad(ind_frag)))**2

    error = np.sqrt(error/n_const)
    return error


# Outer loop -- iterations of QBE

# Inner loop is over different pairs of neighboring/overlapping fragments
    # VQE or Phase estimation call
    # Lagrange update -- gradient descent ++
    # Update the Hamiltonian

def qbe_solver(fragment: qbe_fragment, optimizer_options=None):
    """
    Inputs:
        fragment:
        optimizer_options:
            n_iters: number of iterations
            LR_init: initial learning rate
            MAX_LS:
            THRES_GRAD: Threshold on gradient

    :return:
    """
    # Optimizer
    def_optimizer_options = {'n_iters': 10, 'LR_init': 1.0, 'MAX_LS': 100, 'THRES_GRAD': 1e-7}

    if optimizer_options is None:
        optimizer_options = def_optimizer_options

    n_iters = optimizer_options['n_iters']
    LR_init = optimizer_options['LR_init']
    MAX_LS = optimizer_options['MAX_LS']
    THRES_GRAD = optimizer_options['THRES_GRAD']

    # create data structs
    gradlmbd = np.zeros((n_iters, fragment.n_frag, 2 * fragment.nao, 2 * fragment.nao))  # gradient of the lagragian multipliers
    lmbd = np.zeros((n_iters, fragment.n_frag, 2 * fragment.nao, 2 * fragment.nao))  # lagragian multipliers

    # begin iteration
    error_list = np.zeros(n_iters)
    for ind_iter in range(n_iters):
        for ind_frag in range(fragment.n_frag):

            print("\n Iteration %d, Fragment %d" % (ind_iter, ind_frag))

            # use favorate optimization algorithm to update the lagrangian multiplier
            # perform a line search for example
            print("Start line search:")
            LR = LR_init
            gradlmbd[ind_iter][ind_frag] = fragment.calc_grad(ind_frag)
            dlmbd = gradlmbd[ind_iter][ind_frag]
            dham = lmbd2ham(dlmbd, fragment.nao)
            oldgradnorm = linalg.norm(gradlmbd[ind_iter][ind_frag])

            for isearch in range(MAX_LS):
                # print(gradlmbd[it][frag])
                fragment.qubit_hamiltonians[ind_frag] += LR * dham
                lmbd[ind_iter][ind_frag] += LR * dlmbd

                fragment.f1rdms[ind_frag] = fragment.frag2f1rdm(ind_frag)
                gradlmbd[ind_iter][ind_frag] = fragment.calc_grad(ind_frag)

                gradnorm = linalg.norm(gradlmbd[ind_iter][ind_frag])
                print("---\n%d\t%12.8e" % (isearch, gradnorm))

                if gradnorm < THRES_GRAD:
                    print("Line search converged at grad = %12.8e after %d steps." % (gradnorm, isearch))
                    break
                elif gradnorm > oldgradnorm:
                    fragment.qubit_hamiltonians[ind_frag] = fragment.qubit_hamiltonians[ind_frag] - LR * dham
                    lmbd[ind_iter][ind_frag] = lmbd[ind_iter][ind_frag] - LR * dlmbd
                    LR = 0.5 * LR
                    print("Norm of gradient increases. Rescaling Learning rate = %12.6f." % LR)
                    continue
                else:
                    oldgradnorm = gradnorm
                    dlmbd = gradlmbd[ind_iter][ind_frag]
                    dham = lmbd2ham(dlmbd, fragment.nao)
                    LR = LR_init

                if isearch == MAX_LS - 1:
                    print("Line search does not converge after %d steps. Continue to next fragment." % MAX_LS)

        # Compute the total RMS error for this iteration
        error_list[ind_iter] = tot_err(fragment)
        print("Total error for this iteration is %12.8e." % error_list[ind_iter])

        return error_list


class qbe_solver_qubit(object):
    def __init__(self, fragment: qbe_fragment_qubit, type_constraint='quadratic',
                 type_gs_solver='classical', gs_solver=None, optimizer_options=None):
        """
        Inputs:
            type_gs_solver: ['vqe', 'classical']
            gs_solver: gs_energy, gs_vec = f(n_qubits, hamiltonian, ansatz)

        TODO: need to ensure that if 'vqe' was chosen then 'vqe' was also run during the ground state computation in qbe_init
        """
        # Optimizer
        def_optimizer_options = {'max_iters': 100, 'LR_init': 1.0, 'LR_schedule': None,
                                 'n_gd_iters': 100, 'THRES_GRAD': 1e-8}

        if optimizer_options is None:
            optimizer_options = def_optimizer_options

        self.max_iters = optimizer_options['max_iters']
        self.LR_init = optimizer_options['LR_init']
        self.LR_schedule = optimizer_options['LR_schedule']
        self.n_gd_iters = optimizer_options['n_gd_iters']
        self.THRES_GRAD = optimizer_options['THRES_GRAD']

        self.qbe_frag = fragment

        # Assuming 1 overlapping site at the moment!
        self.delta_lambda = {}
        self.gradients = {}

        self.type_constraint = type_constraint
        for f in fragment.labels:
            # Single element for quadratic constraints and 1d array of len [qbe_frag.overlap_P] for linear case
            self.delta_lambda.update({f: {nb_f: self.matching_constraint(f, nb_f) for nb_f in fragment.neighbors[f]} })
            self.gradients.update({f: {nb_f: self.matching_constraint(f, nb_f) for nb_f in fragment.neighbors[f]} })

        # average of the various gradients of the lagrangian wrt the different lagrange multipliers
        self.norm_gradients = []
        self.norm_gradients.append(self.compute_norm_gradient())

        # RMSE error in fragments -- delta_rho (should be same as above when type_constraints='quadratic')
        self.rmse_error = []
        self.rmse_error.append(self.rmse_fragment_rho())

        # Number of cumulative eigensolver calls
        self.n_eigensolver_calls = []
        self.n_eigensolver_calls.append(0)

        # Cumulative runtime that we keep track of for every big iteration
        self.run_time = []
        self.run_time.append(0.0)

        # Information about Ground state solver
        if type_gs_solver not in ['vqe', 'classical']:
            raise RuntimeError('Illegal input of type of GS solver')

        self.type_gs_solver = type_gs_solver

        if gs_solver is None and type_gs_solver == 'vqe':
            raise ValueError('Require a VQE solver to be inputted')

        self.gs_solver = gs_solver

        # More setup for VQE
        if self.type_gs_solver == 'vqe':
            self.ansatz_param_ic = {}
            for f in fragment.labels:
                # Current initial conditions for ansatz parameters
                self.ansatz_param_ic.update({f: {nb_f: None for nb_f in fragment.neighbors[f]} })

    def compute_norm_gradient(self):
        norm_gradient = 0
        counter = 0
        for f in self.qbe_frag.labels:
            for nb_f in self.qbe_frag.neighbors[f]:
                norm_gradient += np.mean(np.abs(self.gradients[f][nb_f]))
                counter += 1

        return norm_gradient/counter

    def matching_constraint(self, f, nb_f):
        """
        :param f:
        :param nb_f:
        :return:

        TODO: The linear case is actually more complicated, need to add
        """
        if self.type_constraint == 'quadratic':
            delta_lambda = 0
            for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
                rho_E_temp = self.qbe_frag.rho_E[f][nb_f][ind_site]   # rho_E_A with B
                rho_C_temp = self.qbe_frag.rho_C[nb_f][f][ind_site]   # rho_C_B with A

                delta_lambda += 2*np.trace(rho_E_temp @ rho_C_temp) - np.trace(rho_E_temp @ rho_E_temp) - \
                   np.trace(rho_C_temp @ rho_C_temp)

            return delta_lambda

        elif self.type_constraint == 'linear':
            delta_lambda = 0
            for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
                delta_lambda += self.qbe_frag.sigma_rho_E[f][nb_f][ind_site] - self.qbe_frag.sigma_rho_C[nb_f][f][ind_site]

            return delta_lambda
        else:
            raise RuntimeError("Check type of constraint")

    def rmse_fragment_rho(self):
        """
        Note this is very similar to matching constraint -- quadratic case but keeping as different function
        """
        rmse_error = 0.0
        N_sites = 0

        # This used to be computed earlier in a very similar manner to the quadratic constraint
        # for f in self.qbe_frag.labels:
        #     for nb_f in self.qbe_frag.neighbors[f]:
        #         for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
        #             rho_E_temp = self.qbe_frag.rho_E[f][nb_f][ind_site]  # rho_E_A with B
        #             rho_C_temp = self.qbe_frag.rho_C[nb_f][f][ind_site]  # rho_C_B with A
        #
        #             rmse_error += 2 * np.trace(rho_E_temp @ rho_C_temp) - np.trace(rho_E_temp @ rho_E_temp) - \
        #                             np.trace(rho_C_temp @ rho_C_temp)
        #
        #             N_sites += 1

        for f in self.qbe_frag.labels:
            for nb_f in self.qbe_frag.neighbors[f]:
                for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
                    rho_E_temp = self.qbe_frag.rho_E[f][nb_f][ind_site]  # rho_E_A with B
                    rho_C_temp = self.qbe_frag.rho_C[nb_f][f][ind_site]  # rho_C_B with A

                    rmse_error += np.trace( (rho_E_temp - rho_C_temp).H @ (rho_E_temp - rho_C_temp) )
                    N_sites += 1

        rmse_error = np.sqrt(rmse_error/N_sites)

        return np.real(rmse_error)

    def construct_update_matrix(self, rho, site):
        """
        construct the matrix which will be added to the existing fragment hamiltonian during a gradient descent iteration
        This matrix is equal to (rho_CB-rho_EA) tensor product identity

        TODO: Check if the expressions below work for n_qubits_overlapping_site neq 1
        """
        edge_qubit_index = site[0]
        n_qubits_rho = self.qbe_frag.n_qubits_overlapping_site

        qubit_index = self.qbe_frag.n_qubits - 1
        if edge_qubit_index == qubit_index:
            update_matrix = rho
            qubit_index -= n_qubits_rho
        else:
            update_matrix = np.eye(2**n_qubits_rho)
            qubit_index -= 1

        while qubit_index >= 0:
            if qubit_index == edge_qubit_index:
                update_matrix = np.kron(update_matrix, rho)
                qubit_index -= n_qubits_rho
            else:
                update_matrix = np.kron(update_matrix, np.eye(2**n_qubits_rho))
                qubit_index -= 1

        return update_matrix

    def update_fragment_hamiltonian_quadratic(self, f, nb_f, delta_lambda, lr):
        """
        after computing the gradient, complete the gradient descent step by updating the fragment hamiltonian with a perturbation

        Assumes that our dictionaries are "ordered"
        Same delta_lambda is used for all the matrices at the overlapping sites
        """
        dH_f = np.zeros(shape=self.qbe_frag.H[f].shape, dtype=complex)

        for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
            overlapping_site = self.qbe_frag.neighbors[f][nb_f]['edge'][ind_site]

            # (I \otimes (rho_C_B with A - rho_E_A with B)
            dH_f += lr*2*delta_lambda*self.construct_update_matrix(self.qbe_frag.rho_C[nb_f][f][ind_site] -
                                                                   self.qbe_frag.rho_E[f][nb_f][ind_site],
                                                                   overlapping_site)

        self.qbe_frag.H[f] += dH_f

    def update_fragment_hamiltonian_linear(self, f, nb_f, delta_lambda, lr):
        """
        after computing the gradient, complete the gradient descent step by updating the fragment hamiltonian with a perturbation

        Assumes that our dictionaries are "ordered"
        Same delta_lambda is used for all the matrices at the overlapping sites
        """
        dH_f = np.zeros(shape=self.qbe_frag.H[f].shape, dtype=complex)

        for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
            overlapping_site = self.qbe_frag.neighbors[f][nb_f]['edge'][ind_site]

            for ind_P in range(self.qbe_frag.overlap_P['n']):
                P = self.qbe_frag.overlap_P['pauli_mat'][ind_P]
                dH_f += lr * delta_lambda[ind_P] * self.construct_update_matrix(P, overlapping_site)

        self.qbe_frag.H[f] += dH_f

    def gd_solve(self, FLAG_verbose=False, FLAG_logger=False, log_filename='log_file.txt'):
        n_iter = 0
        n_eig_solver_calls = 0
        LR_init = self.LR_init
        start_time = time.perf_counter()

        if FLAG_logger:
            f_log = open(log_filename, "a+")
            # Iteration, Eigensolver Calls, Gradient info, RMSE error in Fragment Rhos, Runtime
            f_log.write("%d %d %3.18f %3.18f %f \n" % (n_iter, n_eig_solver_calls,
                                                       self.norm_gradients[-1], self.rmse_error[-1],
                                                       self.run_time[-1]))
            f_log.close()

        while self.norm_gradients[-1] > self.THRES_GRAD and n_iter < self.max_iters:
            # main loop through all the fragments, matching each fragment to those overlapping with it
            if n_iter % 1 == 0:
                print('Iter. number:', n_iter)

            # loop through all the fragments
            for f in self.qbe_frag.labels:
                # complete num_iter iterations of gradient descent when matching the fragment to its neighbor
                for n in range(self.n_gd_iters):
                    # reduce the learning rate at each step -- should this not also be a function of the run?
                    lr = LR_init * (1.0 - n/self.n_gd_iters)

                    # loop through all the neighboring fragments which overlap with the fragment of interest
                    for nb_f in self.qbe_frag.neighbors[f]:
                        # gradient of the lagrangian with respect to this lagrange multiplier -- array in linear case
                        delta_lambda = self.delta_lambda[f][nb_f]

                        # update the Hamiltonian--this is the gradient descent update
                        if self.type_constraint == 'quadratic':
                            self.update_fragment_hamiltonian_quadratic(f, nb_f, delta_lambda, lr)
                        elif self.type_constraint == 'linear':
                            self.update_fragment_hamiltonian_linear(f, nb_f, delta_lambda, lr)
                        else:
                            raise RuntimeError("Wrong constraint type being worked with")

                        # compute the new ground state using the updated Hamiltonian -- also update rho_E_f, rho_C_f
                        if self.type_gs_solver == 'classical':
                            self.qbe_frag.update_ground_state([f])
                        elif self.type_gs_solver == 'vqe':
                            gs_energy, gs_vec, ansatz_params = self.gs_solver.compute_ground_state(self.qbe_frag.H[f],
                                                                ansatz_parameters_init=self.ansatz_param_ic[f][nb_f])

                            self.qbe_frag.ground_state[f] = DensityMatrix(gs_vec)
                            self.qbe_frag.gs_energies[f] = gs_energy
                            self.ansatz_param_ic[f][nb_f] = copy.deepcopy(ansatz_params)

                        self.qbe_frag.update_partial_rho_fragment([f])

                        n_eig_solver_calls += 1

                        # Update delta_lambda
                        self.delta_lambda[f][nb_f] = self.matching_constraint(f, nb_f)

                        # record the gradient after the final step of gradient descent
                        if n == self.n_gd_iters - 1:
                            self.gradients[f][nb_f] = delta_lambda

                    if FLAG_verbose:
                        print('Updated fragment %d, gd iter. %d, big iter. %d' % (int(f), n, n_iter))

            # compute the average of the gradients for a given run (through the main loop)
            n_iter += 1
            self.norm_gradients.append(self.compute_norm_gradient())

            # Compute RMSE error in fragment rhos
            self.rmse_error.append(self.rmse_fragment_rho())

            self.n_eigensolver_calls.append(n_eig_solver_calls)

            current_time = time.perf_counter()
            self.run_time.append(current_time-start_time)

            if FLAG_logger:
                f_log = open(log_filename, "a+")
                # Iteration, Eigensolver Calls, Gradient info, RMSE error in Fragment Rhos, Runtime
                f_log.write("%d %d %3.18f %3.18f %f \n" % (n_iter, n_eig_solver_calls,
                                                           self.norm_gradients[-1], self.rmse_error[-1],
                                                           self.run_time[-1]))
                f_log.close()

            # Update learning rate
            if self.LR_schedule is not None:
                LR_init = self.LR_schedule(self.LR_init, n_iter)

        ds = {'iterations': n_iter, 'norm_gradients': self.norm_gradients,
              'rmse_error_fragment_rho': self.rmse_error, 'run_time': self.run_time,
              'n_eig_calls': self.n_eigensolver_calls}

        return ds


class qbe_solver_qubit_pd(object):
    def __init__(self, fragment: qbe_fragment_qubit, type_constraint='quadratic',
                 type_gs_solver='classical', gs_solver=None, optimizer_options=None):
        """
        Inputs:
            type_gs_solver: ['vqe', 'classical']
            gs_solver: gs_energy, gs_vec = f(n_qubits, hamiltonian, ansatz)

        TODO: To be merged with above later
        TODO: need to ensure that if 'vqe' was chosen then 'vqe' was also run during the ground state computation in qbe_init
        """
        # Optimizer
        def_optimizer_options = {'max_iters': 100, 'LR_init': 1.0, 'LR_schedule': None,
                                 'n_gd_iters': 100, 'THRES_GRAD': 1e-8}

        if optimizer_options is None:
            optimizer_options = def_optimizer_options

        self.max_iters = optimizer_options['max_iters']
        self.LR_init = optimizer_options['LR_init']
        self.LR_schedule = optimizer_options['LR_schedule']
        self.n_gd_iters = optimizer_options['n_gd_iters']
        self.THRES_GRAD = optimizer_options['THRES_GRAD']

        self.qbe_frag = fragment

        # Assuming 1 overlapping site at the moment!
        self.delta_lambda = {}
        self.gradients = {}

        self.type_constraint = type_constraint
        for f in fragment.labels:
            # Single element for quadratic constraints and 1d array of len [qbe_frag.overlap_P] for linear case
            self.delta_lambda.update({f: {nb_f: self.matching_constraint(f, nb_f) for nb_f in fragment.neighbors[f]} })
            self.gradients.update({f: {nb_f: self.matching_constraint(f, nb_f) for nb_f in fragment.neighbors[f]} })

        # average of the various gradients of the lagrangian wrt the different lagrange multipliers
        self.norm_gradients = []
        self.norm_gradients.append(self.compute_norm_gradient())

        # RMSE error in fragments -- delta_rho (should be same as above when type_constraints='quadratic')
        self.rmse_error = []
        self.rmse_error.append(self.rmse_fragment_rho())

        # Number of cumulative eigensolver calls
        self.n_eigensolver_calls = []
        self.n_eigensolver_calls.append(0)

        # Cumulative runtime that we keep track of for every big iteration
        self.run_time = []
        self.run_time.append(0.0)

        # Information about Ground state solver
        if type_gs_solver not in ['vqe', 'classical']:
            raise RuntimeError('Illegal input of type of GS solver')

        self.type_gs_solver = type_gs_solver

        if gs_solver is None and type_gs_solver == 'vqe':
            raise ValueError('Require a VQE solver to be inputted')

        self.gs_solver = gs_solver

        # More setup for VQE
        if self.type_gs_solver == 'vqe':
            self.ansatz_param_ic = {}
            for f in fragment.labels:
                # Current initial conditions for ansatz parameters
                self.ansatz_param_ic.update({f: {nb_f: None for nb_f in fragment.neighbors[f]} })

        # For diagnosis
        self.updated_paulis = {}
        for f in fragment.labels:
            # Current initial conditions for ansatz parameters
            self.updated_paulis.update(
                {f: {nb_f: {iter: [] for iter in range(self.max_iters)} for nb_f in fragment.neighbors[f]}})

    def compute_norm_gradient(self):
        norm_gradient = 0
        counter = 0
        for f in self.qbe_frag.labels:
            for nb_f in self.qbe_frag.neighbors[f]:
                norm_gradient += np.mean(np.abs(self.gradients[f][nb_f]))
                counter += 1

        return norm_gradient/counter

    def matching_constraint(self, f, nb_f):
        """
        :param f:
        :param nb_f:
        :return:

        TODO: The linear case is actually more complicated, need to add
        """
        if self.type_constraint == 'quadratic':
            delta_lambda = 0
            for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
                rho_E_temp = self.qbe_frag.rho_E[f][nb_f][ind_site]   # rho_E_A with B
                rho_C_temp = self.qbe_frag.rho_C[nb_f][f][ind_site]   # rho_C_B with A

                delta_lambda += 2*np.trace(rho_E_temp @ rho_C_temp) - np.trace(rho_E_temp @ rho_E_temp) - \
                   np.trace(rho_C_temp @ rho_C_temp)

            return delta_lambda

        elif self.type_constraint == 'linear':
            delta_lambda = 0
            for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
                delta_lambda += self.qbe_frag.sigma_rho_E[f][nb_f][ind_site] - self.qbe_frag.sigma_rho_C[nb_f][f][ind_site]

            return delta_lambda
        else:
            raise RuntimeError("Check type of constraint")

    def rmse_fragment_rho(self):
        """
        Note this is very similar to matching constraint -- quadratic case but keeping as different function
        """
        rmse_error = 0.0
        N_const = 0
        for f in self.qbe_frag.labels:
            for nb_f in self.qbe_frag.neighbors[f]:
                for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
                    rho_E_temp = self.qbe_frag.rho_E[f][nb_f][ind_site]  # rho_E_A with B
                    rho_C_temp = self.qbe_frag.rho_C[nb_f][f][ind_site]  # rho_C_B with A

                    rmse_error += 2 * np.trace(rho_E_temp @ rho_C_temp) - np.trace(rho_E_temp @ rho_E_temp) - \
                                    np.trace(rho_C_temp @ rho_C_temp)

                    N_const += 1

        rmse_error = np.sqrt(np.abs(rmse_error)/N_const)

        return rmse_error

    def construct_update_pauli(self, rho, site):
        """
        construct the matrix which will be added to the existing fragment hamiltonian during a gradient descent iteration
        This matrix is equal to (rho_CB-rho_EA) tensor product identity

        TODO: Check if the expressions below work for n_qubits_overlapping_site neq 1
        """
        edge_qubit_index = site[0]
        n_qubits_rho = self.qbe_frag.n_qubits_overlapping_site

        qubit_index = self.qbe_frag.n_qubits - 1
        if edge_qubit_index == qubit_index:
            update_matrix = rho
            qubit_index -= n_qubits_rho
        else:
            update_matrix = 'I'
            qubit_index -= 1

        while qubit_index >= 0:
            if qubit_index == edge_qubit_index:
                update_matrix += rho
                qubit_index -= n_qubits_rho
            else:
                update_matrix += 'I'
                qubit_index -= 1

        return update_matrix

    # def construct_update_matrix(self, rho, site):
    #     """
    #     construct the matrix which will be added to the existing fragment hamiltonian during a gradient descent iteration
    #     This matrix is equal to (rho_CB-rho_EA) tensor product identity
    #
    #     TODO: Check if the expressions below work for n_qubits_overlapping_site neq 1
    #     """
    #     edge_qubit_index = site[0]
    #     n_qubits_rho = self.qbe_frag.n_qubits_overlapping_site
    #
    #     qubit_index = self.qbe_frag.n_qubits - 1
    #     if edge_qubit_index == qubit_index:
    #         update_matrix = rho
    #         qubit_index -= n_qubits_rho
    #     else:
    #         update_matrix = np.eye(2**n_qubits_rho)
    #         qubit_index -= 1
    #
    #     while qubit_index >= 0:
    #         if qubit_index == edge_qubit_index:
    #             update_matrix = np.kron(update_matrix, rho)
    #             qubit_index -= n_qubits_rho
    #         else:
    #             update_matrix = np.kron(update_matrix, np.eye(2**n_qubits_rho))
    #             qubit_index -= 1
    #
    #     return update_matrix
    #
    # def update_fragment_hamiltonian_quadratic(self, f, nb_f, delta_lambda, lr):
    #     """
    #     after computing the gradient, complete the gradient descent step by updating the fragment hamiltonian with a perturbation
    #
    #     Assumes that our dictionaries are "ordered"
    #     Same delta_lambda is used for all the matrices at the overlapping sites
    #     """
    #     dH_f = np.zeros(shape=self.qbe_frag.H[f].shape, dtype=complex)
    #
    #     for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
    #         overlapping_site = self.qbe_frag.neighbors[f][nb_f]['edge'][ind_site]
    #
    #         # (I \otimes (rho_C_B with A - rho_E_A with B)
    #         dH_f += lr*2*delta_lambda*self.construct_update_matrix(self.qbe_frag.rho_C[nb_f][f][ind_site] -
    #                                                                self.qbe_frag.rho_E[f][nb_f][ind_site],
    #                                                                overlapping_site)
    #
    #     self.qbe_frag.H[f] += dH_f
    #
    # def update_fragment_hamiltonian_linear(self, f, nb_f, delta_lambda, lr):
    #     """
    #     after computing the gradient, complete the gradient descent step by updating the fragment hamiltonian with a perturbation
    #
    #     Assumes that our dictionaries are "ordered"
    #     Same delta_lambda is used for all the matrices at the overlapping sites
    #     """
    #     dH_f = np.zeros(shape=self.qbe_frag.H[f].shape, dtype=complex)
    #
    #     for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
    #         overlapping_site = self.qbe_frag.neighbors[f][nb_f]['edge'][ind_site]
    #
    #         for ind_P in range(self.qbe_frag.overlap_P['n']):
    #             P = self.qbe_frag.overlap_P['pauli_mat'][ind_P]
    #             dH_f += lr * delta_lambda[ind_P] * self.construct_update_matrix(P, overlapping_site)
    #
    #     self.qbe_frag.H[f] += dH_f

    def update_fragment_hamiltonian_paulisumop_linear(self, f, nb_f, delta_lambda, lr, n_iter):
        """
        after computing the gradient, complete the gradient descent step by updating the fragment hamiltonian with a perturbation

        Assumes that our dictionaries are "ordered"
        Same delta_lambda is used for all the matrices at the overlapping sites
        """
        dH_f = np.zeros(shape=self.qbe_frag.H[f].shape, dtype=complex)

        for ind_site in range(self.qbe_frag.neighbors[f][nb_f]['n_sites']):
            overlapping_site = self.qbe_frag.neighbors[f][nb_f]['edge'][ind_site]

            for ind_P in range(self.qbe_frag.overlap_P['n']):
                if not np.isclose(np.abs(delta_lambda[ind_P]), 0):
                    P = self.qbe_frag.overlap_P['pauli_strings'][ind_P]

                    # Update the PauliSumOp, target observables and coeffs as required
                    # Get the Pauli string corresponding to update
                    pauli_string_update = self.construct_update_pauli(P, overlapping_site)
                    coeff = lr * delta_lambda[ind_P]

                    if pauli_string_update in self.qbe_frag.target_H[f]:
                        # target_H[f] is a dictionary
                        self.qbe_frag.target_H[f][pauli_string_update] += coeff
                    else:
                        # Add pauli_string_update to target observables
                        self.qbe_frag.target_H[f].update({pauli_string_update: coeff})

                    # Update pauli info -- tracker
                    self.updated_paulis[f][nb_f][n_iter].append((pauli_string_update, coeff))

                    # # Also update the matrix for consistency for now
                    # P_mat = self.qbe_frag.overlap_P['pauli_mat'][ind_P]
                    # dH_f += coeff * self.construct_update_matrix(P_mat, overlapping_site)

        # self.qbe_frag.H[f] += dH_f

        # Reconstruct the PauliSumOp for now (the add() method doesn't update existing Paulis but adds a new one)
        self.qbe_frag.paulisumop_H[f] = PauliSumOp.from_list(
            list([[P, self.qbe_frag.target_H[f][P]] for P in self.qbe_frag.target_H[f]]))

        # Compute Hamiltonian from the PauliSumOp to make sure we are okay in the above computations
        self.qbe_frag.H[f] = copy.deepcopy(self.qbe_frag.paulisumop_H[f].to_spmatrix())

    def gd_solve(self, FLAG_verbose=False, FLAG_logger=False, log_filename='log_file.txt'):
        n_iter = 0
        n_eig_solver_calls = 0
        LR_init = self.LR_init
        start_time = time.perf_counter()

        if FLAG_logger:
            f_log = open(log_filename, "a+")
            # Iteration, Eigensolver Calls, Gradient info, RMSE error in Fragment Rhos, Runtime
            f_log.write("%d %d %3.18f %3.18f %f \n" % (n_iter, n_eig_solver_calls,
                                                       self.norm_gradients[-1], self.rmse_error[-1],
                                                       self.run_time[-1]))
            f_log.close()

        while self.norm_gradients[-1] > self.THRES_GRAD and n_iter < self.max_iters:
            # main loop through all the fragments, matching each fragment to those overlapping with it
            if n_iter % 1 == 0:
                print('Iter. number:', n_iter)

            # loop through all the fragments
            for f in self.qbe_frag.labels:
                # complete num_iter iterations of gradient descent when matching the fragment to its neighbor
                for n in range(self.n_gd_iters):
                    # reduce the learning rate at each step -- should this not also be a function of the run?
                    lr = LR_init * (1.0 - n/self.n_gd_iters)

                    # loop through all the neighboring fragments which overlap with the fragment of interest
                    for nb_f in self.qbe_frag.neighbors[f]:
                        print('(f,nb_f,n_iter) = (%s, %s, %d)' % (f, nb_f, n))

                        # gradient of the lagrangian with respect to this lagrange multiplier -- array in linear case
                        delta_lambda = self.delta_lambda[f][nb_f]

                        # Just linear for now
                        self.update_fragment_hamiltonian_paulisumop_linear(f, nb_f, delta_lambda, lr, n_iter)

                        # compute the new ground state using the updated Hamiltonian -- also update rho_E_f, rho_C_f
                        if self.type_gs_solver == 'classical':
                            self.qbe_frag.update_ground_state([f])
                        elif self.type_gs_solver == 'vqe':
                            gs_energy, gs_vec, ansatz_params = \
                                self.gs_solver.compute_ground_state(self.qbe_frag.paulisumop_H[f],
                                                                    ansatz_parameters_init=self.ansatz_param_ic[f][nb_f],
                                                                    FLAG_pauli_decomposition_instance=True)

                            self.qbe_frag.ground_state[f] = DensityMatrix(gs_vec)
                            self.qbe_frag.gs_energies[f] = gs_energy
                            self.ansatz_param_ic[f][nb_f] = copy.deepcopy(ansatz_params)

                        self.qbe_frag.update_partial_rho_fragment([f])

                        n_eig_solver_calls += 1

                        # Update delta_lambda
                        self.delta_lambda[f][nb_f] = self.matching_constraint(f, nb_f)

                        # record the gradient after the final step of gradient descent
                        if n == self.n_gd_iters - 1:
                            self.gradients[f][nb_f] = delta_lambda

                    if FLAG_verbose:
                        print('Updated fragment %d, gd iter. %d, big iter. %d' % (int(f), n, n_iter))

            # compute the average of the gradients for a given run (through the main loop)
            n_iter += 1
            self.norm_gradients.append(self.compute_norm_gradient())

            # Compute RMSE error in fragment rhos
            self.rmse_error.append(self.rmse_fragment_rho())

            self.n_eigensolver_calls.append(n_eig_solver_calls)

            current_time = time.perf_counter()
            self.run_time.append(current_time-start_time)

            if FLAG_logger:
                f_log = open(log_filename, "a+")
                # Iteration, Eigensolver Calls, Gradient info, RMSE error in Fragment Rhos, Runtime
                f_log.write("%d %d %3.18f %3.18f %f \n" % (n_iter, n_eig_solver_calls,
                                                           self.norm_gradients[-1], self.rmse_error[-1],
                                                           self.run_time[-1]))
                f_log.close()

            # Update learning rate
            if self.LR_schedule is not None:
                LR_init = self.LR_schedule(self.LR_init, n_iter)

        ds = {'iterations': n_iter, 'norm_gradients': self.norm_gradients,
              'rmse_error_fragment_rho': self.rmse_error, 'run_time': self.run_time,
              'n_eig_calls': self.n_eigensolver_calls, 'updated_paulis': self.updated_paulis}

        return ds