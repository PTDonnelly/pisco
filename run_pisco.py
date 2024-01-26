import argparse
import os
from pisco import Extractor, Preprocessor, Processor

def _clean_up_files(ex: Extractor, metop: str):
    
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
        except Exception as e:
            print(f"Error moving file: {e}")

    # Move .sh and .log files
    move_file(source_sh, target_sh)
    move_file(source_log, target_log)


def preprocess_iasi(ex: Extractor, data_level: str):
    """
    This function is used to process IASI (Infrared Atmospheric Sounding Interferometer) data 
    by extracting raw binary files and preprocessing them into pandas DataFrames.

    It first uses an Extractor object to fetch the raw data files for a specific data level, 
    and then performs extraction. 

    If the extraction is successful, the function creates a Preprocessor object using the path 
    to the extracted intermediate file and the data level. This Preprocessor object preprocesses 
    the files to create a DataFrame for the given date.

    Parameters:
    ex (Extractor): An Extractor object which contains methods and attributes for data extraction.
    data_level (str): The data level string, which determines the level of data to be extracted 
                      and preprocessed.

    Returns:
    None: The function performs extraction and preprocessing operations but does not return a value.
    """
    # Use OBR to extract IASI data from raw binary files
    ex.data_level = data_level
    ex.get_datapaths()
    ex.extract_files()

    # If IASI data was successfully extracted
    if ex.intermediate_file_check:
        # Preprocess the data into pandas DataFrames
        pre = Preprocessor(ex)
        
        if ex.config.output_format == "bin":
            pre.preprocess_binary_files()
        elif ex.config.output_format == "txt":
            pre.preprocess_text_files()
        return


def process_iasi(ex: Extractor):
    """
    Correlate level 1C spectra and level 2 cloud products.

    Compares processed IASI products from CSV files and
    stores all spectra co-located with instances of a given Cloud Phase.

    Args:
        Instance of the Extractor class 
    
    Result:
        A CSV file containing all spectra at those locations and times.
    """
    # If IASI data was successfully extracted
    if ex.intermediate_file_check:
        # Instantiate a Processor class 
        pro = Processor(ex)

        # Check that both L1C and L2 data exist
        if pro.check_l1c_l2_data_exist():
            # Merge data sets
            pro.merge_spectra_and_cloud_products()
    return


def run_pisco(metop, year, month, day, config):
    ex = Extractor(config)
    ex.year = f"{year:04d}"
    ex.month = f"{month:02d}"
    ex.day = f"{day:02d}"

    # The Logging context manager (in this location in run_pisco.py) is not needed here due to SLURM's built-in logging functionality 
    if ex.config.L1C:
        preprocess_iasi(ex, data_level="l1c")
    if ex.config.L2:
        preprocess_iasi(ex, data_level="l2")
    if ex.config.process:
        process_iasi(ex)
    
    # Move SLURM script and log file to desired location
    _clean_up_files(ex, metop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IASI data for a given date.")
    parser.add_argument("metop", type=str, help="Satellite identifier")
    parser.add_argument("year", type=int, help="Year to process")
    parser.add_argument("month", type=int, help="Month to process")
    parser.add_argument("day", type=int, help="Day to process")
    parser.add_argument("config", type=str, help="Path to configuration file")

    args = parser.parse_args()
    run_pisco(args.metop, args.year, args.month, args.day, args.config)