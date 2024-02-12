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
    """PISCO: Package for IASI Spectra and Cloud Observations. Main script for launching the IASI data extraction and processing pipeline.

    This script serves as the entry point for running the complete workflow,
    which includes data extraction, processing, and post-processing steps for 
    IASI (Infrared Atmospheric Sounding Interferometer) data.
    It utilizes the `Configurer` class to initialize settings from a JSON configuration file,
    the `Extractor` and `Preprocessor` classes to extract data based on those settings, and
    the `Processor` class to process the extracted data.

    The workflow is as follows:
    1. Configuration: Reads and applies settings from a specified JSON configuration file, including data paths, satellite identifiers, and processing parameters.
    2. Data Extraction: Extracts Level 1C (L1C) or Level 2 (L2) data for the specified date range and satellite, based on the configuration.
    3. Data Processing: Processes the extracted data, potentially including cleaning, merging, and reducing data fields.

    Dependencies:
    - commentjson: Used for loading the configuration file that may include comments.
    - os: Used for file and directory operations.
    - subprocess: Used in the `Extractor` class to run external commands for data extraction.
    - pandas, numpy: Used for data manipulation and analysis.

    Usage:
    To run the script, ensure that a valid configuration file is in place and execute the script from the command line.
    The configuration file path can be set within the script or specified as a command-line argument. The default use case
    can be seen below, where pisco is executed on a day-by-day basis.
    
    If the value of submit_job in the config.json is `true`, the script will automatically submit the jobs to the SLURM
    job scheduler for each day (make sure to adapt the generate_slurm_script above).
    
    If the value of submit_job in the config.json is `false`, then it will generate the same script but will not execute them, allowing the user
    to execute one-by-one if they desire (useful for testing).
    
    If one wishes to simple execute pisco directly on the command line, one needs to specify the necessary arguments to build and execute the function.
    
    The code is currently optimised for SLURM submission, if another use case is desired, feel free to fork the repository and adapt to your needs.
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
