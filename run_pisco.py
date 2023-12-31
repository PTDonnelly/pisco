import os
import cProfile

from pisco import Extractor, Logger, Profiler, scripts

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
                
                # Use Logger as a context manager to capture stdout output to a log file
                with Logger(f"{ex.config.datapath_out}pisco.log") as log:
                    # Depending on the configuration, perform different data processing tasks
                    if ex.config.L1C:
                        valid_indices = scripts.flag_data(ex, data_level="l1c")
                        scripts.preprocess_iasi(ex, valid_indices, data_level="l1c")
                    if ex.config.L2:
                        valid_indices = scripts.flag_data(ex, data_level="l2")
                        scripts.preprocess_iasi(ex, valid_indices, data_level="l2")
                    if ex.config.process:
                        scripts.process_iasi(ex)

                # After the data processing tasks are done, move the log file to the desired location
                os.replace(f"{ex.config.datapath_out}pisco.log", f"{ex.datapath_out}pisco.log")


if __name__ == "__main__":
    # profiler = Profiler()
    # profiler.start()

    main()

    # profiler.stop()
    # profiler.dump_stats('/data/pdonnelly/iasi/cProfiler_output.txt')