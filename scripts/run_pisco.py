import argparse
import logging

from pisco import Extractor, Preprocessor, Processor

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    # Use IASI readers to extract IASI data from raw binary files
    ex.data_level = data_level
    ex.get_datapaths()
    
    if ex.data_level == 'l1c':
        # Run OBR from date parameters
        ex.extract_files()
    elif ex.data_level == 'l2':
        # Scan raw datafiles in the date directory
        file_paths = ex.get_l2_product_files()
        for file_path in file_paths:
            ex.extract_files(file_path)


    # If IASI data was successfully extracted
    if ex.intermediate_file_check:
        # Preprocess the data into pandas DataFrames
        preprocessor = Preprocessor(ex, memory)
        
        # Read OBR textfiles and store to pandas DataFrame
        preprocessor.open_text_file()
        
        if preprocessor.data_level == "l1c":
            # Rename the spectral columns to contain "Spectrum"
            preprocessor.fix_spectrum_columns()
        
        # Construct Local Time column
        preprocessor.build_local_time()
        
        # Construct Datetime column and remove individual time elements
        preprocessor.build_datetime()
        
        # Save filtered DataFrame to compressed pickle
        preprocessor.save_observations()
        
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
    processor = Processor(ex)

    # Check that both L1C and L2 data exist
    if processor.check_l1c_l2_data_exist():
        # Load IASI spectra and cloud products
        processor.load_data()

        # Merge DataFrames, dropping uncorrelated rows and unwanted columns
        processor.combine_datasets()

        # Save merged and filtered DataFrame to compressed pickle
        processor.save_merged_products(delete_intermediate_files=False)
    
    return


def run_pisco(memory, year, month, day):
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
    
    logging.info("Pisco processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IASI data for a given date.")
    parser.add_argument("mem", type=int, help="Memory Request")
    parser.add_argument("year", type=int, help="Year to process")
    parser.add_argument("month", type=int, help="Month to process")
    parser.add_argument("day", type=int, help="Day to process")

    args = parser.parse_args()
    run_pisco(args.mem, args.year, args.month, args.day)