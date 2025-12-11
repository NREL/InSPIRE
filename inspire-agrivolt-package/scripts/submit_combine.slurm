#!/bin/bash
#SBATCH --output=scripts/logs/agrivolt-irr-combine-%x-%j.out
#SBATCH --error=scripts/logs/agrivolt-irr-combine-%x-%j.err
#SBATCH --time=15:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=33
#SBATCH --mem=192G
#SBATCH --account=inspire
#SBATCH --mail-user=tobin.ford@nrel.gov
#SBATCH --mail-type=ALL

source scripts/_env_utils.sh
ENV_NAME="$(get_inspire_agrivolt_conda_env)"

module load anaconda3
source activate "$ENV_NAME"
echo "environment activated"

echo "calling combine.py"
srun python scripts/combine.py