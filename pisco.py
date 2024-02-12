import subprocess
from pisco import Extractor

def generate_slurm_script(metop, year, month, day):
    # Memory request (in GB, used later for optimal file reading)
    mem = 8
    
    # Format date integers to date strings
    year, month, day = f"{year:04d}", f"{month:02d}", f"{day:02d}"
    
    # Prepare SLURM submission script
    script_name = f"/data/pdonnelly/iasi/pisco_{metop}_{year}_{month}_{day}.sh"
    script_content = f"""#!/bin/bash
#SBATCH --job-name=pisco_{metop}_{year}_{month}_{day}
#SBATCH --output=/data/pdonnelly/iasi/pisco_{metop}_{year}_{month}_{day}.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem={mem}GB

# Purge all modules to prevent conflict with current environnement
module purge

# Load necessary modules
module load python/meso-3.8

python /data/pdonnelly/github/pisco/scripts/run_pisco.py {mem} {metop} {year} {month} {day}

"""
    with open(script_name, 'w') as file:
        file.write(script_content)

    return script_name

def main():
    """PISCO: Package for IASI Spectra and Cloud Observations

    For each date specified, open raw binary files, reduce into intermediate files using optimised C scripts
    developed by IASI team, then produce conveniently-formatted spatio-temporal data
    of IASI products: L1C calibrated spectra or L2 cloud products.
    """
    # Instantiate an Extractor class to get data from raw binary files
    ex = Extractor()
    
    # The MetOp satellite identifier for these observations (metopa, metopb, or metopc)
    metop = ex.config.satellite_identifier

    # Scan years, months, days (specific days or all calendar days, dependent on Config attributes)
    for year in ex.config.year_list:
        month_range = ex.config.month_list if (not ex.config.month_list == "all") else range(1, 13)
        
        for im, month in enumerate(month_range):
            day_range = ex.config.day_list if (not ex.config.day_list == "all") else range(1, ex.config.days_in_months[month-1] + 1)
            
            for day in day_range:
                script = generate_slurm_script(metop, year, month, day)
                
                # Set execute permissions on the script
                subprocess.run(["chmod", "+x", script])

                if ex.config.submit_job:
                    # Submit the batch script to SLURM using sbatch
                    subprocess.run(["sbatch", script])
                # else:
                #     # Run on command line on login node
                #     subprocess.run([script])

if __name__ == "__main__":
    main()