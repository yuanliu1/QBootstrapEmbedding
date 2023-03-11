#!/bin/bash

# Initialize the module command first
source /etc/profile

# Load Conda environment
conda activate qbe-qiskit

echo "My run number: " $1

# Call your script as you would from the command line passing $1 and $2 as arguments
#python ToyModel_runner_vqe.py --run_seed=$1 --expt_id=1
#python H4_runner_vqe.py --run_seed=$1 --expt_id=2
python H4_runner_no_LO2MO_vqe.py --run_seed=$1 --expt_id=2
