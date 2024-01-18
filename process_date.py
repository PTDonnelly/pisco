import argparse
import os
from pisco import Extractor, Logger, scripts

def process_date(year, month, day, config):
    ex = Extractor(config)
    ex.year = year
    ex.month = month
    ex.day = day

    with Logger(f"{ex.config.datapath_out}pisco_{ex.year}_{ex.month}_{ex.day}.log") as log:
        if ex.config.L1C:
            valid_indices = scripts.flag_data(ex, data_level="l1c")
            scripts.preprocess_iasi(ex, valid_indices, data_level="l1c")
        if ex.config.L2:
            valid_indices = scripts.flag_data(ex, data_level="l2")
            scripts.preprocess_iasi(ex, valid_indices, data_level="l2")
        if ex.config.process:
            scripts.process_iasi(ex)
    
    # Move SLURM script and log file to desired location
    os.replace(f"{ex.config.datapath_out}pisco_{ex.year}_{ex.month}_{ex.day}.sh", f"{ex.datapath_out}pisco_{ex.year}_{ex.month}_{ex.day}.sh")
    os.replace(f"{ex.config.datapath_out}pisco_{ex.year}_{ex.month}_{ex.day}.log", f"{ex.datapath_out}pisco_{ex.year}_{ex.month}_{ex.day}.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IASI data for a given date.")
    parser.add_argument("metop", type=str, help="Satellite identifier")
    parser.add_argument("year", type=int, help="Year to process")
    parser.add_argument("month", type=int, help="Month to process")
    parser.add_argument("day", type=int, help="Day to process")
    parser.add_argument("config", type=str, help="Path to configuration file")

    args = parser.parse_args()
    process_date(args.metop, args.year, args.month, args.day, args.config)