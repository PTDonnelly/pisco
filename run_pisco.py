import argparse
import os
from pisco import Extractor, Preprocessor, Processor

def flag_data(ex: Extractor, data_level: str):
    """This function is a truncated form of the main preprocess_iasi function, but bypasses most
    of the functionality. It creates a much smaller dataset with OBR, scans it and returns the indices 
    of observations within the specified latitude-longitude range.
    """
    
    # Use OBR to extract IASI data from raw binary files
    if data_level == "l1c":
        ex.config.channels = ex.config.set_channels("flag")
    ex.data_level = data_level
    ex.get_datapaths()
    ex.extract_files()

    # If IASI data was successfully extracted
    if ex.intermediate_file_check:
        # Preprocess the data into pandas DataFrames
        pre = Preprocessor(ex.intermediate_file, ex.data_level, ex.config.latitude_range, ex.config.longitude_range)
        
        # Open binary file and extract metadata
        pre.open_binary_file()

        # Limit observations to specified spatial range
        valid_indices = pre.flag_observations_to_keep(pre.metadata._get_iasi_common_record_fields())

        # Closed binary file and extract metadata
        pre.close_binary_file()

        return valid_indices
    return


def preprocess_iasi(ex: Extractor, valid_indices: np.array, data_level: str):
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
    if data_level == "l1c":
        ex.config.channels = ex.config.set_channels("range")
    ex.data_level = data_level
    ex.get_datapaths()
    ex.extract_files()

    # If IASI data was successfully extracted
    if ex.intermediate_file_check:
        # Preprocess the data into pandas DataFrames
        pre = Preprocessor(ex.intermediate_file, ex.data_level, ex.config.latitude_range, ex.config.longitude_range)
        pre.preprocess_files(ex.year, ex.month, ex.day, valid_indices)
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
    # Instantiate a Processor class 
    pro = Processor(ex.config.datapath_out, ex.year, ex.month, ex.day, ex.config.cloud_phase)

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
        valid_indices = flag_data(ex, data_level="l1c")
        preprocess_iasi(ex, valid_indices, data_level="l1c")
    if ex.config.L2:
        valid_indices = flag_data(ex, data_level="l2")
        preprocess_iasi(ex, valid_indices, data_level="l2")
    if ex.config.process:
        process_iasi(ex)
    
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