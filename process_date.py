import os
from pisco import Extractor, Logger, scripts

def process_date(year, month, day, config):
    ex = Extractor(config)
    ex.year = f"{year:04d}"
    ex.month = f"{month:02d}"
    ex.day = f"{day:02d}"

    with Logger(f"{ex.config.datapath_out}pisco_{year}_{month}_{day}.log") as log:
        if ex.config.L1C:
            valid_indices = scripts.flag_data(ex, data_level="l1c")
            scripts.preprocess_iasi(ex, valid_indices, data_level="l1c")
        if ex.config.L2:
            valid_indices = scripts.flag_data(ex, data_level="l2")
            scripts.preprocess_iasi(ex, valid_indices, data_level="l2")
        if ex.config.process:
            scripts.process_iasi(ex)
    
    # Move SLURM script and log file to desired location
    os.replace(f"{ex.config.datapath_out}pisco_{year}_{month}_{day}.sh", f"{ex.datapath_out}pisco_{year}_{month}_{day}.sh")
    os.replace(f"{ex.config.datapath_out}pisco_{year}_{month}_{day}.log", f"{ex.datapath_out}pisco_{year}_{month}_{day}.log")