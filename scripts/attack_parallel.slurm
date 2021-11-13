#!/bin/bash

#SBATCH --nodes 1
#SBATCH --job-name="MD_attack"
#SBATCH --output=slurm-%A-%x.out
#SBATCH --account="punim0784"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time 72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=curtishuang@live.com

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi


# Run the job from this directory:
cd /data/gpfs/projects/punim0784/hanxunh/MD_attacks

# The modules to load:
source /usr/local/module/spartan_new.sh
module load web_proxy
module load gcc/8.3.0 cuda/10.1.243 openmpi/3.1.4
module load tensorflow/1.15.2-python-3.7.4

# Attack
defence=$1
attack=$2

python main.py --defence $defence --attack $attack \
               --bs 100 --data_parallel
