import logging
import os

from pisco import Extractor

# Obtain a logger for this module
logger = logging.getLogger(__name__)

def move_job_files(ex: Extractor, metop: str):
    
    # Define the output file name
    output_file = f"pisco_{metop}_{ex.year}_{ex.month}_{ex.day}"

    # Define source and target paths for the .sh and .log files
    source_sh = os.path.join(ex.config.datapath, f"{output_file}.sh")
    target_sh = os.path.join(ex.config.datapath, metop, f"{ex.year}", f"{ex.month}", f"{ex.day}", f"{output_file}.sh")

    source_log = os.path.join(ex.config.datapath, f"{output_file}.log")
    target_log = os.path.join(ex.config.datapath, metop, f"{ex.year}", f"{ex.month}", f"{ex.day}", f"{output_file}.log")

    # Function to move a file
    def move_file(source, target):
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)  # Ensure target directory exists
            os.replace(source, target)
            logger.info(f"Moved job file to: {output_file}")
        except Exception as e:
            logger.error(f"Error moving file: {e}")

    # Move .sh and .log files
    move_file(source_sh, target_sh)
    move_file(source_log, target_log)