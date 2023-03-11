#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arkopal@mit.edu

#conda init bash
eval "$(conda shell.bash hook)"
conda activate qbe-qiskit

python H8_runner_vqe_LO2MO.py --type_constraints='linear' --expt_id=1
