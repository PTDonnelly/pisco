import logging

from pisco import Configurer
from scripts import prepare_job_submission as job

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    # Instantiate an Configurer to create output directories and generate job files
    config = Configurer()

    # Scan years, months, days (specific days or all calendar days, dependent on Config attributes)
    for year in config.year_list:
        month_range = config.month_list if (not config.month_list == "all") else range(1, 13)
        
        for im, month in enumerate(month_range):
            day_range = config.day_list if (not config.day_list == "all") else range(1, config.days_in_months[month-1] + 1)
            
            for day in day_range:
                # Create output directory
                output_path = job.create_output_directory(config.datapath, config.satellite_identifier, year, month, day)

                # Check if there is already a complete run, if not execute on this day
                run_exists =  job.check_pisco_log(output_path)

                if not run_exists:
                    # Create SLURM shell script and log file, and place them in the output folder
                    script_name = job.create_job_file(output_path, year, month, day)
                    print(output_path)
                    # if config.submit_job:
                    #     # Submit the batch script to SLURM using sbatch and capture the last job ID
                    #     job_id = job.submit_job_file(output_path, script_name)
                    #     if job_id:
                    #         # Update last_job_id with the latest submitted job ID
                    #         last_job_id = job_id

    # # After all processing jobs are submitted, submit a cleanup job with dependency
    # if config.submit_job:
    #     job.cleanup_job_files(config.datapath, last_job_id)

if __name__ == "__main__":
    main()
