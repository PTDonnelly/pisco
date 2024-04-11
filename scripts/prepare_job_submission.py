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
    except OSError as e:
        logging.error(f"Error creating directory: {output_path}: {e}")
    return output_path


def check_pisco_log(output_path):
        """
        Checks if 'pisco.log' exists in the specified output directory and contains
        the string "Pisco processing complete.". If the condition is met, it means
        a processing run has been completed, and returns False. Otherwise, returns True
        indicating that the processing should proceed.

        :param output_path: Path to the output directory where 'pisco.log' is expected.
        :return: False if 'pisco.log' exists and contains "Pisco processing complete.",
                 True otherwise.
        """
        log_file_path = os.path.join(output_path, 'pisco.log')
        print(log_file_path)
        try:
            with open(log_file_path, 'r') as log_file:
                for line in log_file:
                    print(line)
                    if "Pisco processing complete." in line:
                        input()
                        return False
        except FileNotFoundError:
            # If 'pisco.log' does not exist, processing should proceed
            pass

        # If 'pisco.log' does not contain the completion string or does not exist, return True
        return True


def create_job_file(output_path: str, year: str, month: str, day: str) -> str:
    year, month, day = format_date_elements(year, month, day)
    
    # Memory request (in GB, used later for optimal file reading)
    allocated_memory = 8

    # Prepare SLURM submission script
    script_name = f"{output_path}/pisco.sh"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=pisco
#SBATCH --output={output_path}/pisco.log
#SBATCH --time=01:00:00
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


def submit_job_file(output_path: str, script_name: str) -> str:
    try:
        # Submit the batch script to SLURM using sbatch and capture the output
        result = subprocess.run(["sbatch", script_name], capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            # Log the standard output if the command succeeded
            logging.info(f"{result.stdout.strip()} at {output_path}")
            # Extract and return the job ID
            job_id = result.stdout.strip().split()[-1]  # Assumes the job ID is the last element
            return job_id
        else:
            # Log the standard error if the command failed
            logging.error(f"Failed to submit batch script: {result.stderr.strip()}")
            return None

    except subprocess.CalledProcessError as e:
        # Log the exception if the subprocess call raised an error
        logging.error(f"An error occurred while submitting the batch script: {e}")
        return None
    

def cleanup_job_files(datapath: str, last_job_id: str) -> None:
    script_name = "cleanup_job.sh"

    script_content = f"""#!/bin/bash
#SBATCH --job-name=pisco
#SBATCH --output={datapath}/pisco_cleanup.log
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --mem=1GB

# Purge all modules to prevent conflict with current environnement
module purge

# Navigate to the datapath directory
cd {datapath}

# Check if cd command was successful
if [ $? -ne 0 ]; then
    echo "Failed to navigate to datapath: {datapath}"
    exit 1
fi

# Recursively delete all empty directories within datapath
echo "Deleting all empty directories within {datapath}..."
find . -type d -empty -delete

# Log completion message
echo "Cleanup of empty directories completed successfully at $(date)"
"""
    with open(script_name, 'w') as file:
        file.write(script_content)

    result = subprocess.run(["sbatch", "--dependency=afterok:" + last_job_id, script_name], capture_output=True, text=True)
    
    # Check if the command was successful
    if result.returncode == 0:
        # Log the standard output if the command succeeded
        logging.info(f"Batch clean-up script submitted on job ID {last_job_id} successfully: {result.stdout.strip()}")
    else:
        # Log the standard error if the command failed
        logging.error(f"Failed to submit batch clean-up script: {result.stderr.strip()}")
    return None