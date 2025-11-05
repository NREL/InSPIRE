#!/bin/bash
#SBATCH --output=logs/agrivolt-irr-combine-check-%x-%j.out
#SBATCH --error=logs/agrivolt-irr-combine-check-%x-%j.err
#SBATCH --time=15:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=150G
#SBATCH --account=inspire
#SBATCH --mail-user=tobin.ford@nrel.gov
#SBATCH --mail-type=ALL

module load anaconda3
source activate /home/tford/.conda-envs/geospatial

export PYTHONUNBUFFERED=1
srun -u python /home/tford/dev/InSPIRE/inspire-agrivolt-package/scripts/check_combine.py