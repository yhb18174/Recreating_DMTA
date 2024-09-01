#!/bin/bash

#======================================================
#
# Job script for running a parallel job on a single node
#
#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=standard
#
# Specify project account
#SBATCH --account=palmer-addnm
#
# No. of tasks required (max. of 40) (1 for a serial job)
#SBATCH --ntasks=40
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=168:00:000
#
# Job name
#SBATCH --job-name=10_mp
#
# Output file
#SBATCH --output=slurm-%j.out
#=======================================================


module purge

module load anaconda/python-3.9.7

#=========================================================
# Prologue script to record job details
# Do not change the line below
#=========================================================
/opt/software/scripts/job_prologue.sh 
#----------------------------------------------------------

# Modify the line below to run your program
source activate phd_env

n_cmpds=10
sel_method="mp"
start_iter=1
total_iters=1

python -u /users/yhb18174/Recreating_DMTA/scripts/run/run_DMTA.py $n_cmpds $sel_method $start_iter $total_iters

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
if [ -f /opt/software/scripts/job_epilogue.sh ]
then
    /opt/software/scripts/job_epilogue.sh
fi
#----------------------------------------------------------