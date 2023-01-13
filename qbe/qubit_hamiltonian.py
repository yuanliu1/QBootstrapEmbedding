# Convert the Fermionic H to the qubit H

# Diagonalize the qubit H to figure out the "true" ground state -- classical solve

def fham2qham(h1, h2):
    # inumpyut the 1e and 2e integrals in MO orbitals, output the corresponding dense matrix in qubit format

    nao = h1.shape[0]
    electronic_energy_from_ints = ElectronicEnergy.from_raw_integrals(
        ElectronicBasis.MO, h1, h2.reshape((nao, nao, nao, nao))
    )
    # print(electronic_energy_from_ints)

    ferOp = electronic_energy_from_ints.second_q_ops()[0]  # here, output length is always 1
    qubit_converter = QubitConverter(mapper=JordanWignerMapper())
    qubitOp = qubit_converter.convert(ferOp)  # this will not give two-qubit reduction
    # qubitOp = qubit_converter.convert(ferOp, nao) # use this and the above to get two-qubit reduction
    # print(qubitOp)
    # print(qubitOp.num_qubits)
    ham = qubitOp.to_matrix()
    return ham