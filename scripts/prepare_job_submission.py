import logging
import os
import subprocess
from typing import Tuple

def format_date_elements(year: int, month: int, day: int) -> Tuple[str, str, str]:
    return (f"{year:04d}", f"{month:02d}", f"{day:02d}")

def create_output_directory(datapath: str, satellite_identifier: str, year: str, month: str, day: str) -> str:
    year, month, day = format_date_elements(year, month, day)
    output_path = os.path.join(datapath, satellite_identifier, year, month, day)
    try:
        os.makedirs(output_path, exist_ok=True)
        logging.info(f"Created directory: {output_path}")
    except OSError as e:
        logging.error(f"Error creating directory: {output_path}: {e}")
    return output_path


def create_job_file(output_path: str, year: str, month: str, day: str) -> str:
    year, month, day = format_date_elements(year, month, day)
    
    # Memory request (in GB, used later for optimal file reading)
    allocated_memory = 8

    # Prepare SLURM submission script
    script_name = f"{output_path}/pisco.sh"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=pisco
#SBATCH --output={output_path}/pisco.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem={allocated_memory}GB

# Purge all modules to prevent conflict with current environnement
module purge

# Load necessary modules
module load python/meso-3.8

python /data/pdonnelly/github/pisco/scripts/run_pisco.py {allocated_memory} {year} {month} {day}

"""
    with open(script_name, 'w') as file:
        file.write(script_content)

    # Set execute permissions on the script
    subprocess.run(["chmod", "+x", script_name])
        
    return script_name


def submit_job_file(script_name: str) -> None:
    try:
        # Submit the batch script to SLURM using sbatch and capture the output
        result = subprocess.run(["sbatch", script_name], capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            # Log the standard output if the command succeeded
            logging.info(f"Batch script submitted successfully: {result.stdout.strip()}")
        else:
            # Log the standard error if the command failed
            logging.error(f"Failed to submit batch script: {result.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        # Log the exception if the subprocess call raised an error
        logging.error(f"An error occurred while submitting the batch script: {e}")