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
#SBATCH --time=168:00:00
#
# Job name
#SBATCH --job-name=50_mp
#
# Output file
#SBATCH --output=slurm-%j.out

## Email settings
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=huw.williams.2018@uni.strath.ac.uk
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

n_cmpds=50
sel_method="mp"
start_iter=5
total_iters=26
run_date="20240916"

python -u /users/yhb18174/Recreating_DMTA/scripts/run/run_DMTA.py $n_cmpds $sel_method $start_iter $total_iters $run_date

#=========================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#=========================================================
if [ -f /opt/software/scripts/job_epilogue.sh ]
then
    /opt/software/scripts/job_epilogue.sh
fi
#----------------------------------------------------------