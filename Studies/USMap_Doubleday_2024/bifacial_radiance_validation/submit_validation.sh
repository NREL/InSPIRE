#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17 # we changed this was previously 1, and changed from slurmcluster to localcluster
#SBATCH --account=inspire
#SBATCH --mail-user=kate.doubleday@nrel.gov
#SBATCH --mail-type=ALL

module load anaconda3
. activate radianceEnv
module load gcc
source addPaths.sh
python bifacial_radiance_comparison.py