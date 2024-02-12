
from pisco import Extractor, Preprocessor

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