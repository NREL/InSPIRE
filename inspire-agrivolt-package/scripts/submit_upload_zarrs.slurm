#!/bin/bash
#SBATCH --output=scratch/tford/agrivolt-irr-deploy-%x-%j.out
#SBATCH --error=scratch/tford/agrivolt-irr-deploy-%x-%j.err
#SBATCH --time=15:00:00
#SBATCH --partition=standard
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --account=inspire
#SBATCH --mail-user=tobin.ford@nrel.gov
#SBATCH --mail-type=ALL

set -euo pipefail

module load anaconda3
source activate /home/tford/.conda-envs/boto

for item in "03.zarr" "04.zarr" "05.zarr" "06.zarr" "07.zarr" "08.zarr" "09.zarr" "10.zarr"; do
    srun --exclusive -N1 -n1 python scripts/upload.py "$item" &
done

wait