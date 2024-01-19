import argparse
import os
from pisco import Extractor, Logger, scripts

def run_pisco(metop, year, month, day, config):
    ex = Extractor(config)
    ex.year = f"{year:04d}"
    ex.month = f"{month:02d}"
    ex.day = f"{day:02d}"

    # The Logging context manager (in this location in run_pisco.py) is not needed here due to SLURM's built-in logging functionality 
    if ex.config.L1C:
        valid_indices = scripts.flag_data(ex, data_level="l1c")
        scripts.preprocess_iasi(ex, valid_indices, data_level="l1c")
    if ex.config.L2:
        valid_indices = scripts.flag_data(ex, data_level="l2")
        scripts.preprocess_iasi(ex, valid_indices, data_level="l2")
    if ex.config.process:
        scripts.process_iasi(ex)
    
    # Move SLURM script and log file to desired location
    output_file = f"pisco_{metop}_{ex.year}_{ex.month}_{ex.day}"
    os.replace(f"{ex.config.datapath}{output_file}.sh", f"{ex.config.datapath}{metop}/merged/{ex.year}/{ex.month}/{ex.day}/{output_file}.sh")
    os.replace(f"{ex.config.datapath}{output_file}.log", f"{ex.config.datapath}{metop}/merged/{ex.year}/{ex.month}/{ex.day}/{output_file}.log")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IASI data for a given date.")
    parser.add_argument("metop", type=str, help="Satellite identifier")
    parser.add_argument("year", type=int, help="Year to process")
    parser.add_argument("month", type=int, help="Month to process")
    parser.add_argument("day", type=int, help="Day to process")
    parser.add_argument("config", type=str, help="Path to configuration file")

    args = parser.parse_args()
    run_pisco(args.metop, args.year, args.month, args.day, args.config)