import gzip
import logging
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, List, Dict, Optional

import pickle

from pisco import Extractor

# Obtain a logger for this module
logger = logging.getLogger(__name__)

class Preprocessor:
    """Handles preprocessing of IASI data for analysis.

    This class is designed to read text file outputs from OBR or generic L2 reader,
    structure the data into a pandas DataFrame,
    and perform data manipulations to prepare it for further analysis.
    It supports handling large datasets by chunking and efficiently manages memory usage during the process.

    Attributes:
        intermediate_file (str): Path to the intermediate file containing raw OBR data.
        delete_intermediate_files (bool): Indicates whether intermediate files should be deleted after processing.
        channels (List[int]): Channels to be included in the processing.
        allocated_memory (int): Memory allocated for processing data, in bytes.
        memory_safety_margin (float): Fraction of allocated memory used as a safety margin during chunking.
        df (pd.DataFrame or None): DataFrame holding processed data, initialized as None.

    Methods:
        calculate_chunk_size(dtype_dict): Calculates optimal chunk size for file reading.
        read_file_in_chunks(dtype_dict): Reads file in calculated chunks and merges into a DataFrame.
        should_load_in_chunks(): Checks if file needs to be loaded in chunks based on size and memory.
        _get_fields_and_datatypes(): Generates a dictionary of field names and their data types for file reading.
        open_text_file(): Opens and reads the intermediate file, either as a whole or in chunks.
        fix_spectrum_columns(): Renames spectrum columns for clarity.
        _calculate_local_time(): Calculates local time to determine day or night.
        build_local_time(): Adds a column indicating day or night based on local time.
        build_datetime(): Combines date and time columns into a single datetime column.
        _delete_intermediate_file(filepath): Deletes the specified intermediate file if required.
        save_observations(delete_intermediate_files=None): Saves processed data to a file and optionally deletes the intermediate file.
    """
    def __init__(self, ex: Extractor, allocated_memory: int, memory_safety_margin=0.5):
        self.intermediate_file: str = ex.intermediate_file_path
        self.delete_intermediate_files = ex.config.delete_intermediate_files
        self.channels: List[int] = ex.channels
        self.allocated_memory = allocated_memory * (1024 ** 3) # Convert from Gigabytes to Bytes
        self.memory_safety_margin = memory_safety_margin
        self.df = None

    def calculate_chunk_size(self, dtype_dict: Dict):       
        # Open the first 100 rows of the csv to check memory usage of DataFrame
        sample_df = pd.read_csv(self.intermediate_file, sep="\t", dtype=dtype_dict, nrows=100)
        memory_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)
        
        # Add safety margin for memory overheads and non-DataFrame memory usage
        available_memory = self.allocated_memory * (1 - self.memory_safety_margin)

        # Calculate chunk size
        chunk_size = int(available_memory / memory_per_row)
        
        logger.info(f"Available memory: {available_memory} B")
        logger.info(f"Memory per row (+50% margin): {memory_per_row} B")
        logger.info(f"Chunk size: {chunk_size} rows")
        return chunk_size


    def read_file_in_chunks(self, dtype_dict: Dict):
        # Load in chunks
        logger.info("Loading in chunks...")
        
        # Initialize a list to hold processed chunks
        chunk_list = []
        
        # Specify the chunk size
        chunk_size = self.calculate_chunk_size(dtype_dict, self.allocated_memory)

        # Iterate over the file in chunks
        for chunk in pd.read_csv(self.intermediate_file, sep="\t", dtype=dtype_dict, chunksize=chunk_size):
            # Append the processed chunk to the list
            chunk_list.append(chunk)

        logger.info(f"Number of chunks: {len(chunk_list)}")

        # Concatenate all processed chunks at once
        concatenated_df = pd.concat(chunk_list, ignore_index=True)
        
        # Ensure all columns have the correct data types
        for column, dtype in dtype_dict.items():
            concatenated_df[column] = concatenated_df[column].astype(dtype)

        return concatenated_df


    def should_load_in_chunks(self) -> bool:
        "Checks if file size is greater than the allocated memory with safety margin"
        file_size = os.path.getsize(self.intermediate_file)
        return file_size > (self.allocated_memory / self.memory_safety_margin)


    def open_text_file(self, ex: Extractor) -> None:
        logger.info("Loading intermediate text file:")
        
        # Create dtype dict from combined fields
        dtype_dict = ex._get_fields_and_datatypes()

        if self.should_load_in_chunks():
            self.df = self.read_file_in_chunks(dtype_dict)
        else:
            # Read in as normal
            self.df = pd.read_csv(self.intermediate_file, sep="\t", dtype=dtype_dict)
        return


    def fix_spectrum_columns(self) -> None:
        # Create a renaming mapping by prepending "Spectrum " to each spectral channel column name
        rename_mapping = {str(channel_id): f"Spectrum {channel_id}" for channel_id in self.channels}

        # Rename the columns using the mapping
        self.df.rename(columns=rename_mapping, inplace=True)
        return


    def select_geographic_region(self, ex: Extractor) -> None:
        # Use ex.config.latitude_range and ex.config.longitude_range to select only the data in that window
        lat_min, lat_max = ex.config.latitude_range
        lon_min, lon_max = ex.config.longitude_range
        
        # Filter the DataFrame for the specified geographic region
        self.df = self.df[(self.df['Latitude'] >= lat_min) & (self.df['Latitude'] <= lat_max) &
                          (self.df['Longitude'] >= lon_min) & (self.df['Longitude'] <= lon_max)]
        return
    
    def build_datetime(self, ex: Extractor) -> List:
        """
        Stores the datetime components to a single column and drops the elements.
        """
        if ex.data_level == "l1c":
            self.df['Datetime'] = pd.to_datetime(
                self.df['Year'].astype(int).astype(str).str.zfill(4) +
                self.df['Month'].astype(int).astype(str).str.zfill(2) +
                self.df['Day'].astype(int).astype(str).str.zfill(2) +
                self.df['Hour'].astype(int).astype(str).str.zfill(2) +
                self.df['Minute'].astype(int).astype(str).str.zfill(2),
                format='%Y%m%d%H%M'
                )
            
        elif ex.data_level == "l2":
            # Ensure both columns are strings for concatenation
            self.df['Date'] = self.df['Date'].astype(str)
            self.df['Time'] = self.df['Time'].astype(str)
            # If 'Time' values are not zero-padded, you might need to pad them
            self.df['Time'] = self.df['Time'].str.pad(width=6, side='left', fillchar='0')
            
            # Concatenate 'Date' and 'Time' columns into a single 'Datetime' string
            self.df['Datetime'] = self.df['Date'] + self.df['Time']
            # Convert 'Datetime' string to a datetime object
            self.df['Datetime'] = pd.to_datetime(self.df['Datetime'], format='%Y%m%d%H%M%S')

            # Drop individual Date and Time columns
            self.df.drop(columns=['Date', 'Time'], inplace=True)
            
            # Extract year, month, day, hour, minute, and milliseconds components (for calculation of Local Time)
            self.df['Year'] = self.df['Datetime'].dt.year
            self.df['Month'] = self.df['Datetime'].dt.month
            self.df['Day'] = self.df['Datetime'].dt.day
            self.df['Hour'] = self.df['Datetime'].dt.hour
            self.df['Minute'] = self.df['Datetime'].dt.minute
            self.df['Milliseconds'] = self.df['Datetime'].dt.microsecond // 1000  # datetime represents of fractional seconds as microseconds)
    
        return 
    

    def _calculate_local_time(self) -> None:
        """
        Calculate the local time (in hours, UTC) that determines whether it is day or night at a specific longitude.

        Returns:
        np.ndarray: Local time (in hours, UTC) within a 24 hour range, used to determine day (6-18) or night (0-6, 18-23).
        """

        # Retrieve the necessary field data
        hour, minute, millisecond, longitude = self.df['Hour'], self.df['Minute'], self.df['Milliseconds'], self.df['Longitude']

        # Calculate the total time in hours, minutes, and milliseconds
        total_time = (hour * 1e4) + (minute * 1e2) + (millisecond / 1e3)

        # Extract the hour, minute and second components from total_time
        hour_component = np.floor(total_time / 10000)
        minute_component = np.mod((total_time - np.mod(total_time, 100)) / 100, 100)
        second_component_in_minutes = np.mod(total_time, 100) / 60

        # Calculate the total time in hours
        total_time_in_hours = (hour_component + minute_component + second_component_in_minutes) / 60

        # Convert longitude to time in hours and add it to the total time
        total_time_with_longitude = total_time_in_hours + (longitude / 15)

        # Add 24 hours to the total time to ensure it is always positive
        total_time_positive = total_time_with_longitude + 24

        # Take the modulus of the total time by 24 to get the time in the range of 0 to 23 hours
        time_in_range = np.mod(total_time_positive, 24)

        # Subtract 6 hours from the total time, shifting the reference for day and night (so that 6 AM becomes 0)
        time_shifted = time_in_range - 6

        # Take the modulus again to ensure the time is within the 0 to 23 hours range
        return np.mod(time_shifted, 24)


    def _format_fractional_time_to_hhmmss(self, fractional_hours):
        """
        Convert fractional hours into a formatted time string HHMMSS.

        Args:
        fractional_hours (np.ndarray): Array of fractional hours.

        Returns:
        np.ndarray: Formatted time strings.
        """
        # Extract the integer part (hours) and fractional part
        hours = np.floor(fractional_hours)
        fractional_minutes = (fractional_hours - hours) * 60
        
        # Repeat for minutes -> seconds
        minutes = np.floor(fractional_minutes)
        seconds = np.round((fractional_minutes - minutes) * 60)

        # Format as strings with zero padding
        formatted_time = np.char.zfill(hours.astype(int).astype(str), 2) + \
                        np.char.zfill(minutes.astype(int).astype(str), 2) + \
                        np.char.zfill(seconds.astype(int).astype(str), 2)

        return formatted_time


    def build_local_time(self) -> List:
        """
        Stores the local time Boolean indicating whether the current time is day or night.
        """
        # Calculate the local time
        local_time = self._calculate_local_time()

        # Store the Boolean indicating day (True) or night (False) in the DataFrame
        self.df['DayNightQualifier'] = (6 < local_time) & (local_time < 18)
        
        # # Convert to formatted time strings
        # formatted_local_time = self._format_fractional_time_to_hhmmss(local_time)
        # self.df['Local Time'] = formatted_local_time

        # Drop original time element columns (in place to save on memory)
        self.df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Milliseconds'], inplace=True)
        return


    def _delete_intermediate_file(self, filepath) -> None:
        """Deletes the specified intermediate file."""
        # If config.delete_intermediate_files is True, try deleting the intermediate OBR data file
        try:
            os.remove(filepath)
            logger.info(f"Deleted intermediate file: {filepath}")

        except OSError as e:
            logger.error(f"Error deleting file: {e}")


    def save_observations(self, delete_intermediate_files: Optional[bool]=None) -> None:
        """
        Saves the observation data to CSV/HDF5 file and deletes OBR output file.
        """  
        # Split the intermediate file path into the root and extension, and give new extension
        file_root, _ = os.path.splitext(self.intermediate_file)
        output_file = file_root + ".pkl.gz"
        
        try:
            # Compress and save using gzip
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(self.df, f)
            
            # Output information on the final DataFrame
            logger.info(self.df.info())
            logger.info(self.df.head())
            logger.info(f"Saved DataFrame to: {output_file}")
        
        except OSError as e:
            logger.error(f"Error saving file: {e}")
        
        # Delete intermediate OBR output file
        if (delete_intermediate_files is None) and self.delete_intermediate_files:
            # If boolean flag is not manually passed, default to the boolean flag in config.delete_intermediate_files
            self._delete_intermediate_file(self.intermediate_file)

        return