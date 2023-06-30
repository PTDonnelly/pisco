import os
import sys
import cProfile

from pisco import Extractor, flag_data, preprocess_iasi, process_iasi

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
    
    # Scan years, months, days (specific days or all calendar days, dependent on Config attributes)
    for year in ex.config.year_list:
        ex.year = f"{year:04d}"

        month_range = ex.config.month_list if (not ex.config.month_list == "all") else range(1, 13)
        for im, month in enumerate(month_range):
            ex.month = f"{month:02d}"

            day_range = ex.config.day_list if (not ex.config.day_list == "all") else range(1, ex.config.days_in_months[im] + 1)
            for day in day_range:
                ex.day = f"{day:02d}"
                
                # Setup output logging to save console output to file                
                original_stdout = sys.stdout # Backup stdout
                sys.stdout = Logger(f"{ex.config.datapath_out}pisco.log") # Replace stdout with Logger class

                if (ex.config.L1C) or (ex.config.L2):
                    valid_indices = flag_data(ex, data_level="l2")
                if ex.config.L1C:
                    preprocess_iasi(ex, valid_indices, data_level="l1c")
                if ex.config.L2:
                    preprocess_iasi(ex, valid_indices, data_level="l2")
                if ex.config.process:
                    process_iasi(ex)
                
                # Move logfile to output directory
                os.replace(f"{ex.config.datapath_out}pisco.log", f"{ex.datapath_out}pisco.log")

                # Restore stdout
                sys.stdout = original_stdout

class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        # It is possible to specify some extra behavior here.
        pass

if __name__ == "__main__":
    import time
    import cProfile
    import io
    import pstats
    # Create profiler
    pr = cProfile.Profile()
    # Start profiler
    pr.enable()

    # Start clock
    start = time.time()

    main()

    # Stop clock
    end = time.time()
    # Print elapsed time
    print(f"Elapsed time: {end-start} s")
    
    # Stop profiler
    pr.disable()
    # Print profiler output to file
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.strip_dirs()
    ps.print_stats()

    with open('/data/pdonnelly/iasi/metopc/cProfiler_output.txt', 'w+') as f:
        f.write(s.getvalue())
