import argparse
import logging
import os

from pisco import Extractor
from scripts.preprocess_iasi import preprocess_iasi
from scripts.process_iasi import process_iasi
from scripts.move_job_files import move_job_files

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pisco(memory, metop, year, month, day):
    """Executes PISCO for a given date, with all settings configured in the config.json.
    """
    # Instantiate an Extractor for this run
    ex = Extractor()
    ex.year = f"{year:04d}"
    ex.month = f"{month:02d}"
    ex.day = f"{day:02d}"

    # The Logging context manager (in this location in run_pisco.py) is not needed here due to SLURM's built-in logging functionality 
    if ex.config.L1C:
        preprocess_iasi(ex, memory, data_level="l1c")
    if ex.config.L2:
        preprocess_iasi(ex, memory, data_level="l2")
    if ex.config.process:
        process_iasi(ex)
    
    # Move SLURM script and log file to desired location
    move_job_files(ex, metop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IASI data for a given date.")
    parser.add_argument("mem", type=int, help="Memory Request")
    parser.add_argument("metop", type=str, help="Satellite identifier")
    parser.add_argument("year", type=int, help="Year to process")
    parser.add_argument("month", type=int, help="Month to process")
    parser.add_argument("day", type=int, help="Day to process")

    args = parser.parse_args()
    run_pisco(args.mem, args.metop, args.year, args.month, args.day)