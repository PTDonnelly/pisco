import os
import subprocess
from pisco import Extractor

def generate_slurm_script(year, month, day, config_file, script_name):
    script_content = f"""#!/bin/bash
#SBATCH --job-name=pisco_{year}_{month}_{day}
#SBATCH --output=/data/pdonnelly/iasi/logfiles/pisco_{year}_{month}_{day}.log
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8GB

# Purge all modules to prevent conflict with current environnement
module purge

# Load necessary modules
module load python/meso-3.8

python process_date.py {year} {month} {day} {config_file}

"""
    
    with open(script_name, 'w') as file:
        file.write(script_content)

def main():
    """PISCO: Package for IASI Spectra and Cloud Observations

    For each date specified, open raw binary files, reduce into intermediate files using optimised C scripts
    developed by IASI team, then produce conveniently-formatted spatio-temporal data
    of IASI products: L1C calibrated spectra or L2 cloud products.
    """
    # Point to location of jsonc configuration file
    path_to_config_file = "./inputs/config.jsonc"
    
    # Instantiate an Extractor class to get data from raw binary files
    ex = Extractor(path_to_config_file)

    # Create the output directory if it doesn't exist
    log_directory = "/data/pdonnelly/iasi/pisco_logfiles/"
    os.makedirs(log_directory, exist_ok=True)

    # Scan years, months, days (specific days or all calendar days, dependent on Config attributes)
    for year in ex.config.year_list:
        month_range = ex.config.month_list if (not ex.config.month_list == "all") else range(1, 13)
        
        for im, month in enumerate(month_range):
            day_range = ex.config.day_list if (not ex.config.day_list == "all") else range(1, ex.config.days_in_months[im] + 1)
            
            for day in day_range:
                script_name = f"{log_directory}pisco_{year}_{month}_{day}.sh"
                generate_slurm_script(year, month, day, path_to_config_file, script_name)
                
                # Set permissions and submit the batch script to SLURM
                subprocess.run(["chmod", "+x", script_name])

if __name__ == "__main__":
    main()
