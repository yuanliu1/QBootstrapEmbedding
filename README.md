# Quantum Bootstrap Embedding (QBE)

## Introduction

In this work, we extend molecular bootstrap embedding to make it appropriate for implementation on a quantum computer.  This enables solution of the electronic structure problem of a large molecule as an optimization problem for a composite Lagrangian governing fragments of the total system, in such a way that fragment solutions can harness the capabilities of quantum computers.  By employing state-of-art quantum subroutines including the quantum SWAP test and quantum amplitude amplification, we show how a quadratic speedup can be obtained over the classical algorithm, in principle. Utilization of quantum computation also allows the algorithm to match -- at little additional computational cost -- full density matrices at fragment boundaries, instead of being limited to 1-RDMs. Current quantum computers are small, but quantum bootstrap embedding provides a potentially generalizable strategy for harnessing such small machines through quantum fragment matching.

## Structure of the Repo

The repo is composed of several files: 
- <code>cases</code>, <code>cases_vqe</code> are scripts that are used to directly produce all calculations in this work. This will be the place to look at for specific implementations of relevant algorithms mentioned in the paper. 
- In addition, <code>data</code> contains all the integrals and Hamiltonian files for the H<sub>8</sub> molecule in the current work. <code>figures</code> contains all plot script/data that can be used to reproduce all figures in the paper. <code>jobs_vqe</code> is an additional folder that has QBE+VQE related scripts. 
- Last but not the least, we have been reorganizing the entire QBE framework into a standalone package in the folder <code>qbe</code>, where an example demonstration of using this codebase is provided in <code>cases/demo_qbe_classical.ipynb</code>.

- __More details on the <code>cases</code> folder__: This folder contains the code and data used to perform the calculation for each individual parts of the QBE algorithm. It contains the following sub-folders or files:
   * <code>linear-qbe_deterministic</code>: scripts to run QBE with linear constraint on H<sub>8</sub> molecules. <code>H8_lin.py</code> is the main code, which read data files and save results in each iteration of QBE to separate files. <code>energy_calc.py</code> is a script to compute the energy from saved ground state vector <code>gs_vec_lin_runxx.pkl</code>  for each run. <code>rmse_H8_lin.npy</code> stores the output root-mean-square-deviation of the mismatch in each run.
   * <code>quadratic-qbe_deterministic</code>: scripts to run QBE with quadratic constraint on H<sub>8</sub> molecules. <code>H8_quad.py</code> is the main code, which read data files and save results in each iteration of QBE to separate files. <code>energy_calc.py</code> is a script to compute the energy from saved ground state vector <code>gs_vec_quad_runxx.pkl</code>  for each run. <code>rmse_H8_quad.npy</code> stores the output root-mean-square-deviation of the mismatch in each run.
   * <code>qpe+swap</code>: Code to perform SWAP test on two H<sub>4</sub> molecules where each one is solved using a quantum phase estimation routine. <code>h4_swap_qpe.py</code> is the main code, and <code>post_process.py</code> compute the standard deviation of the overlap from shot-based simulation results. The subfolder <code>h4_$n$site.py</code> contains the simulation shot count when there is $n$ hydrogen atoms overhap between the two H<sub>4</sub> molecules. <code>integrals</code> folder contains the one- and two-electron integral files used for the two H<sub>4</sub> molecules in the simulation.
   * <code>vmc</code>: <code>bootstrap_embedding_vmc_H8.py</code> is the main script to perform classical bootstrap embedding using variational Monte Carlo as a solver, which serves as benchmark as compared to quantum bootstrap embedding.
   * <code>vqe</code>: QBE using variational quantum eigensolver as fragment solver. <code>H4_runner_vqe.py</code> is QBE+VQE for H<sub>4</sub> molecules, while <code>ToyModel_runner_vqe.py</code> is QBE+VQE for a top model composed of a 4 spin system which is splitted as two fragments with each of them having 3 spins. <code>plot_jobs.ipynb</code> is some plot utility. Different subfolders with name <code>h4_linear_vqe*.ipynb</code> and <code>toymodel_linear_vqe_gd_001</code> are the generated running results for H4 and for the toy model. Note that in all QBE+VQE, only linear constraint is used.

## Requirements
See <code>environment.yml</code> for the dependencies and packages needed to run the codes.

## Citation

Liu, Y., Meitei, O. R., Chin, Z. E., Dutt, A., Tao, M., Chuang, I. L., & Van Voorhis, T.(2023). Bootstrap Embedding on a Quantum Computer. [J. Chem. Theory Comput. 19, 8, 2230–2247 (2023)](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00012). [arXiv:2301.01457](https://arxiv.org/abs/2301.01457)



