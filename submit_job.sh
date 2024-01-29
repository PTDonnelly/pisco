"""#!/bin/bash
#SBATCH --job-name=pisco_plot
#SBATCH --output=/data/pdonnelly/iasi/plot_pisco.log
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8GB

# Purge all modules to prevent conflict with current environnement
module purge

# Load necessary modules
module load python/meso-3.8

python /data/pdonnelly/github/pisco/pisco/scripts.py

"""