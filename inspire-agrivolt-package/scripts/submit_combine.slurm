#!/bin/bash
#SBATCH --output=logs/agrivolt-irr-combine-%x-%j.out
#SBATCH --error=logs/agrivolt-irr-combine-%x-%j.err
#SBATCH --time=15:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=33
#SBATCH --mem=192G
#SBATCH --account=inspire
#SBATCH --mail-user=tobin.ford@nrel.gov
#SBATCH --mail-type=ALL

module load anaconda3
source activate /home/tford/.conda-envs/geospatial

srun python /home/tford/dev/InSPIRE/inspire-agrivolt-package/scripts/combine.py