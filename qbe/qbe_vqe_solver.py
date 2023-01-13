import copy

import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix as scp_coo_matrix
from scipy.sparse import linalg as scp_linalg
from functools import reduce
import itertools

from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector, DensityMatrix

from qiskit.opflow import (PauliOp, SummedOp, PauliExpectation, PauliSumOp,
    OperatorBase,
    ExpectationBase,
    ExpectationFactory,
    StateFn,
    CircuitStateFn,
    ListOp,
    CircuitSampler,
)

from qiskit.algorithms.optimizers import L_BFGS_B, COBYLA, SPSA
from qiskit.algorithms import  MinimumEigensolver, VQE, NumPyMinimumEigensolver
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import TwoLocal


# For a Pauli operator P (specified as a string),
# returns the matrix representation of P as a scipy.sparse.csr_matrix object.
def pauli_to_sparse(P):
    x = ''
    for i in range(len(P)):
        if P[i] == 'I' or P[i] == 'Z':
            x = x + '0'
        else:
            x = x + '1'
    x = int(x, 2)

    z = ''
    for i in range(len(P)):
        if P[i] == 'I' or P[i] == 'X':
            z = z + '0'
        else:
            z = z + '1'
    z = int(z, 2)

    y = 0
    for i in range(len(P)):
        if P[i] == 'Y':
            y += 1

    rows = [r for r in range(2 ** len(P))]

    cols = [r ^ x for r in range(2 ** len(P))]

    vals = []
    for r in range(2 ** len(P)):
        sgn = bin(r & z)
        vals.append(((-1.0) ** sum([int(sgn[i]) for i in range(2, len(sgn))])) * ((-1j) ** y))

    m = scp_coo_matrix((vals, (rows, cols)))

    return m.tocsr()


# Get the mapping to go from qiskit ordering to physics ordering
def get_mapping_qo_to_po(n_qubits):
    """
    Example:
        physics ordering: (1st qubit, 2nd qubit);
        |00> = [1,0,0,0], |01> = [0,1,0,0], |10> = [0,0,1,0], |11> = [0,0,0,1]

        qiskit ordering: (2nd qubit, 1st qubit); wrt above
        |00> = |00>_p = [1,0,0,0]; |01> = |10>_p=[0,0,1,0]; |10> = |01>_p = [0,1,0,0]; |11> = |11>_p = [0,0,0,1]

    To go from qiskit ordering to physics ordering:
        - create list of binary strings in qiskit ordering
        - flip orderings in each list
        - convert each binary string to decimal number
        - find mapping to go to the ascending order

    Notation:
        qo - qiskit ordering
        po - physics ordering
    """
    # create all binary strings
    n_strings = 2**n_qubits

    list_integers_qo = np.arange(2**n_qubits)
    binary_strings_qo = [np.binary_repr(list_integers_qo[ind], width=n_qubits) for ind in range(n_strings)]

    binary_strings_po = [binary_strings_qo[ind][::-1] for ind in range(n_strings)]
    list_integers_po = [int(binary_strings_po[ind], 2) for ind in range(n_strings)]

    mapping_qo_to_po = np.argsort(list_integers_po)

    return mapping_qo_to_po


class QBEVQESolver(object):
    def __init__(self, n_qubits, ansatz=None, backend=None,
                 nshots=10000, seed=170,
                 FLAG_pauli_decomposition_instance=False,
                 FLAG_lo2mo=False, U_lo2mo=None):

        # For computing expecations of Hamiltonians in physics ordering
        if not FLAG_pauli_decomposition_instance:
            single_qubit_paulis = ["I", "X", "Y", "Z"]
            pauli_list = list(itertools.product(single_qubit_paulis, repeat=n_qubits))

            n_P = 4 ** n_qubits
            if n_P != len(pauli_list):
                raise ValueError("Something is up with len(pauli_list)")

            for ind in range(n_P):
                pauli_list[ind] = ''.join(pauli_list[ind])

            # For each, get a scipy matrix (assuming physics ordering)
            matrix_pauli_list = {P: pauli_to_sparse(P) for P in pauli_list}

            self.pauli_list = pauli_list
            self.matrix_pauli_list = matrix_pauli_list

        self.n_qubits = n_qubits
        self.mapping_qo2po = get_mapping_qo_to_po(n_qubits)

        self.ansatz = ansatz
        self.backend = backend
        self.nshots = nshots
        self.seed = seed

        # Unitary operator that takes you from the LO 2 MO basis
        self.FLAG_lo2mo = FLAG_lo2mo
        if FLAG_lo2mo and U_lo2mo is None:
            raise ValueError("Need to input U_lo2mo if corresponding FLAG is on!")

        self.U_lo2mo = U_lo2mo

    def set_ansatz(self, ansatz):
        self.ansatz = ansatz

    def set_backend(self, backend):
        self.backend = backend

    def pauli_decomposition(self, H, FLAG_qiskit_ordering=False, FLAG_filter=True):
        """
        Don't save 0 values if FLAG_filter is on
        """
        if H.shape[0] != 2**self.n_qubits:
            raise RuntimeError("Dimension mismatch between Hamiltonian and QBEVQESolver initialization")

        # Compute the expectation value: alpha_P = Tr(P H)
        if not FLAG_filter:
            if FLAG_qiskit_ordering:
                pd_H_qiskit_ordering = {P[::-1]: (1.0/2**self.n_qubits)*np.trace(self.matrix_pauli_list[P] @ H) for P in self.pauli_list}
            else:
                pd_H_qiskit_ordering = {P: (1.0/2**self.n_qubits)*np.trace(self.matrix_pauli_list[P] @ H) for P in self.pauli_list}
            # pauli_decomposition_H_qiskit_ordering = {P: np.trace(matrix_pauli_list[P] @ H1)  for P in pauli_list}
        else:
            pd_H_qiskit_ordering = {}
            for P in self.pauli_list:
                tr_PH = (1.0/2**self.n_qubits)*np.trace(self.matrix_pauli_list[P] @ H)
                if np.abs(tr_PH) > 1e-16:
                    if FLAG_qiskit_ordering:
                        pd_H_qiskit_ordering.update({P[::-1]: tr_PH})
                    else:
                        pd_H_qiskit_ordering.update({P: tr_PH})

        return pd_H_qiskit_ordering

    def compute_ground_state(self, H, ansatz_parameters_init=None, max_iters=200,
                             FLAG_pauli_decomposition_instance=False,
                             FLAG_qiskit_ordering=False, FLAG_verbose=False, FLAG_debug=False):
        """
        Currently works only with Aer's "statevector_simulator"

        TODO: Allow backend to be aer_simulator and device
        """
        # Convert Hamiltonian to MO basis if desired
        if self.FLAG_lo2mo:
            H_mo = self.U_lo2mo @ H @ self.U_lo2mo.conj().transpose()
            #H_mo = self.U_lo2mo.conj().transpose() @ H @ self.U_lo2mo
        else:
            H_mo = copy.deepcopy(H)

        # Switch to checking instance later
        if FLAG_pauli_decomposition_instance:
            qubit_op_H = copy.deepcopy(H)
        else:
            # Get pauli decomposition of H (in qiskit ordering?)
            pd_H = self.pauli_decomposition(H_mo, FLAG_qiskit_ordering=FLAG_qiskit_ordering, FLAG_filter=True)

            # Convert to PauliSumOp
            list_pauli_values = [(P, pd_H[P]) for P in pd_H]
            qubit_op_H = PauliSumOp.from_list(list_pauli_values)

        # run a vqe solve
        # Ref: https://qiskit.org/documentation/tutorials/algorithms/03_vqe_simulation_with_noise.html
        algorithm_globals.random_seed = self.seed

        qi = QuantumInstance(backend=self.backend, seed_simulator=self.seed, seed_transpiler=self.seed,
                             shots=self.nshots)

        counts = []
        values = []

        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)

        if FLAG_verbose:
            print('VQE solver with seed=%d' % self.seed)

        #opt_method = SPSA(maxiter=max_iters)
        opt_method = L_BFGS_B(maxiter=max_iters)
        vqe = VQE(self.ansatz, optimizer=opt_method, initial_point=ansatz_parameters_init,
                  callback=store_intermediate_result, quantum_instance=qi)

        result = vqe.compute_minimum_eigenvalue(operator=qubit_op_H)
        # result.update({'opt_tracker': {'iters': counts, 'loss': values}})

        if FLAG_verbose:
            print(f'VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}')

        # Convert ground state vector into physics ordering
        if not FLAG_debug:
            if FLAG_qiskit_ordering:
                gs_vec = result.eigenstate[self.mapping_qo2po]
            else:
                gs_vec = result.eigenstate

            gs_energy = result.eigenvalue
            opt_params = list(result.optimal_parameters.values())

            if self.FLAG_lo2mo:
                # Convert back into LO basis
                gs_vec = self.U_lo2mo.conj().transpose() @ gs_vec

            return gs_energy, gs_vec, opt_params
        else:
            return result, counts, values
