#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --account=inspire
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=tobin.ford@nrel.gov


echo "this is a test slurm job to verify email notifications"