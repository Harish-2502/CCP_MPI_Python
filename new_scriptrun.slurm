#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=04:00:00

module purge
module load foss/2022a
module load Python/3.10.4
# module load numpy/1.18.0-python-3.7.4
module load mpi4py/3.1.4

srun python3 new_script.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
