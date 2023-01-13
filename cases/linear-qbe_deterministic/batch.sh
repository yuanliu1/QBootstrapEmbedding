#!/bin/bash
##SBATCH -J H8_energy_calc
#SBATCH -J H8_linear
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=100GB   
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuanliu@mit.edu
#SBATCH -o slurm-%x-%A.out
#SBATCH -e slurm-%x-%A.err

#conda init bash
eval "$(conda shell.bash hook)"
conda activate env_qiskit

export OMP_NUM_THREADS=12
python H8_lin.py
#python energy_calc.py

