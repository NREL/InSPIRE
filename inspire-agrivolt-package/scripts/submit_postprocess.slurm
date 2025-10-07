#!/bin/bash
#SBATCH --output=logs/agrivolt-irr-postprocess-%x-%j.out
#SBATCH --error=logs/agrivolt-irr-postprocess-%x-%j.err
#SBATCH --time=5:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=100G
#SBATCH --account=inspire
#SBATCH --mail-user=tobin.ford@nrel.gov
#SBATCH --mail-type=ALL

module load anaconda3
source activate /home/tford/.conda-envs/geospatial

srun python /home/tford/dev/InSPIRE/Studies/USMap_Doubleday_2024/inspire-agrivolt-package/scripts/postprocess.py
