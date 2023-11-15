#!/bin/bash
#SBATCH --job-name=mipfda
#SBATCH --cpus-per-task=20
#SBATCH --time=0-23:00
#SBATCH --mem=100GB
#SBATCH --output=Z_000.out.log
#SBATCH --error=Z_000.err.log

echo "Current working directory: `pwd`"
echo "Starting run at: `date`"

module load python/3.10.2

srun python Tesina_cluster.py

echo "Job finished with exit code $? at: `date`"
