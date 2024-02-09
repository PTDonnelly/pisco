import argparse
import logging
import os

from pisco import Extractor, Preprocessor, Processor

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def preprocess_iasi(ex: Extractor, memory: int, data_level: str):
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
        pre = Preprocessor(ex, memory)
        
        # Read OBR textfiles and store to pandas DataFrame
        pre.open_text_file()
        if pre.data_level == "l1c":
            # Rename the spectral columns to contain "Spectrum"
            pre.fix_spectrum_columns()
        # Construct Local Time column
        pre.build_local_time()
        # Construct Datetime column and remove individual time elements
        pre.build_datetime()
        # Save filtered DataFrame to compressed pickle
        pre.save_observations()
        
        # Print the DataFrame
        logging.info(pre.df.info(verbose=True))
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
    pro = Processor(ex)

    # Check that both L1C and L2 data exist
    if pro.check_l1c_l2_data_exist():
        # Load IASI spectra and cloud products
        pro.load_data()      
        
        # Correlates measurements, keep matching locations and times of observation
        pro.correlate_datasets()
        
        # Merge DataFrames, dropping uncorrelated rows and unwanted columns
        pro.combine_datasets()

        # Save merged and filtered DataFrame to compressed pickle
        pro.save_merged_products()
        
        logging.info(pro.df.info(verbose=True))
    return


def run_pisco(memory, metop, year, month, day):
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
    _clean_up_files(ex, metop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IASI data for a given date.")
    parser.add_argument("mem", type=int, help="Memory Request")
    parser.add_argument("metop", type=str, help="Satellite identifier")
    parser.add_argument("year", type=int, help="Year to process")
    parser.add_argument("month", type=int, help="Month to process")
    parser.add_argument("day", type=int, help="Day to process")

    args = parser.parse_args()
    run_pisco(args.mem, args.metop, args.year, args.month, args.day)