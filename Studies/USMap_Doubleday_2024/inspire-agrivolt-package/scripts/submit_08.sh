#!/bin/bash
#SBATCH --job-name=agrivolt-irr         # Job name
#SBATCH --output=agrivolt-irr-%j.log    # Standard output (%j for job ID)
#SBATCH --error=agrivolt-irr-%j.err     # Standard error (%j for job ID)
#SBATCH --time=04:00:00                 # Total run time (hh:mm:ss)
#SBATCH --partition=shared              # Queue/partition to use
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks
#SBATCH --cpus-per-task=6               # CPUs per task
#SBATCH --mem=80G                       # Memory for the job
#SBATCH --account=inspire               # Account name

module load anaconda3  # kestrel module name
source activate /home/tford/.conda-envs/geospatial

agrivolt_ground_irradiance Colorado /projects/inspire/PySAM-MAPS/Full-Outputs/Colorado /home/tford/dev/InSPIRE/Studies/USMap_Doubleday_2024/SAM --confs 08 --port 22118 --workers 8

conda deactivate