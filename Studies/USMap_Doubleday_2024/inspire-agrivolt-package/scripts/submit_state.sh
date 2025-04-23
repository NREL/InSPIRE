#!/bin/bash
#SBATCH --output=logs/agrivolt-irr-%x-%j.log
#SBATCH --error=logs/agrivolt-irr-%x-%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --account=inspire

STATE=$1

module load anaconda3
source activate /home/tford/.conda-envs/geospatial

agrivolt_ground_irradiance "$STATE" /projects/inspire/PySAM-MAPS/test-all-states/"$STATE" /home/tford/dev/InSPIRE/Studies/USMap_Doubleday_2024/SAM --confs 01 02 03 04 05 06 07 08 09 10 --port 22118 --workers 8

conda deactivate
