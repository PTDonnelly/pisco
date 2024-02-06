import gzip
import os
import numpy as np
import pandas as pd
from typing import Any, List, BinaryIO, Tuple, List, Optional

import pickle

from pisco import Extractor

class Metadata:
    """
    Metadata class provides the structure and methods to read and process
    metadata of binary files.

    Attributes:
        f (BinaryIO): The binary file object.
        header_size (int): The size of the header in bytes.
        record_header_size (int): The size of the record header.
        number_of_channels (int): The number of channels.
        channel_IDs (np.array): The IDs of the channels.
        record_size (int): The size of each record in bytes.
        number_of_measurements (int): The number of measurements.
    """
    def __init__(self, file: BinaryIO):
        self.f: BinaryIO = file
        self.header_size: int = None
        self.record_size: int = None
        self.number_of_measurements: int = None
        self.number_of_channels: int = None
        self.channel_IDs: np.array = None
        self.number_of_l2_products: int = None
        self.l2_product_IDs: List[int] = None
    
    @staticmethod
    def _get_iasi_common_record_fields() -> List[Tuple]:
        # Format of fields in binary file (field_name, data_type, data_size, cumulative_data_size)
        common_fields = [
                        ('Year', 'uint16', 2, 2),
                        ('Month', 'uint8', 1, 3),
                        ('Day', 'uint8', 1, 4),
                        ('Hour', 'uint8', 1, 5),
                        ('Minute', 'uint8', 1, 6),
                        ('Milliseconds', 'uint32', 4, 10),
                        ('Latitude', 'float32', 4, 14),
                        ('Longitude', 'float32', 4, 18),
                        ('Satellite Zenith Angle', 'float32', 4, 22),
                        ('Bearing', 'float32', 4, 26),
                        ('Solar Zenith Angle', 'float32', 4, 30),
                        ('Solar Azimuth', 'float32', 4, 34),
                        ('Field of View Number', 'uint32', 4, 38),
                        ('Orbit Number', 'uint32', 4, 42),
                        ('Scan Line Number', 'uint32', 4, 46),
                        ('Height of Station', 'float32', 4, 50)]
        return common_fields
  

    @staticmethod
    def _get_iasi_l1c_record_fields() -> List[Tuple]:
        # Format of general L1C-specific fields in binary file (field_name, data_type, data_size, cumulative_data_size),
        # cumulative total continues from the fourth digit of the last tuple in common_fields.
        l1c_fields = [
                    ('Day version', 'uint16', 2, 2 + offset),
                    ('Start Channel 1', 'uint32', 4, 6 + offset),
                    ('End Channel 1', 'uint32', 4, 10 + offset),
                    ('Quality Flag 1', 'uint32', 4, 14 + offset),
                    ('Start Channel 2', 'uint32', 4, 18 + offset),
                    ('End Channel 2', 'uint32', 4, 22 + offset),
                    ('Quality Flag 2', 'uint32', 4, 26 + offset),
                    ('Start Channel 3', 'uint32', 4, 30 + offset),
                    ('End Channel 3', 'uint32', 4, 34 + offset),
                    ('Quality Flag 3', 'uint32', 4, 38 + offset),
                    ('Cloud Fraction', 'uint32', 4, 42 + offset),
                    ('Surface Type', 'uint8', 1, 43 + offset)]
        return l1c_fields
        
    @staticmethod
    def _get_iasi_l2_record_fields() -> List[Tuple]:
       # Format of general L2-specific fields in binary file (field_name, data_type, data_size, cumulative_data_size),
        # cumulative total continues from the fourth digit of the last tuple in common_fields.
        l2_fields = [
                    ('Superadiabatic Indicator', 'uint8', 1, 1 + offset),
                    ('Land Sea Qualifier', 'uint8', 1, 2 + offset),
                    ('Day Night Qualifier', 'uint8', 1, 3 + offset),
                    ('Processing Technique', 'uint32', 4, 7 + offset),
                    ('Sun Glint Indicator', 'uint8', 1, 8 + offset),
                    ('Cloud Formation and Height Assignment', 'uint32', 4, 12 + offset),
                    ('Instrument Detecting Clouds', 'uint32', 4, 16 + offset),
                    ('Validation Flag for IASI L1 Product', 'uint32', 4, 20 + offset),
                    ('Quality Completeness of Retrieval', 'uint32', 4, 24 + offset),
                    ('Retrieval Choice Indicator', 'uint32', 4, 28 + offset),
                    ('Satellite Manoeuvre Indicator', 'uint32', 4, 32 + offset)]
        return l2_fields

    @staticmethod
    def _get_l2_product_fields(product: int, offset: int=0) -> List[Tuple]:
        # Format of fields in binary file (field_name, data_type, data_size, cumulative_data_size)
        if product == "clp":
            fields = [
                    ('Vertical Significance', 'uint32', 4, 4 + offset),
                    ('Pressure 1', 'float32', 4, 8 + offset),
                    ('Temperature or Dry Bulb Temperature 1', 'float32', 4, 12 + offset),
                    ('Cloud Amount in Segment 1', 'float32', 4, 16 + offset),
                    ('Cloud Phase 1', 'uint32', 4, 20 + offset),
                    ('Pressure 2', 'float32', 4, 24 + offset),
                    ('Temperature or Dry Bulb Temperature 2', 'float32', 4, 28 + offset),
                    ('Cloud Amount in Segment 2', 'float32', 4, 32 + offset),
                    ('Cloud Phase 2', 'uint32', 4, 36 + offset),
                    ('Pressure 3', 'float32', 4, 40 + offset),
                    ('Temperature or Dry Bulb Temperature 3', 'float32', 4, 44 + offset),
                    ('Cloud Amount in Segment 3', 'float32', 4, 48 + offset),
                    ('Cloud Phase 3', 'uint32', 4, 52 + offset)
                    ]
        if product == "twt":
            fields = []
        if product == "ozo":
            fields = [
                    ('Selection Background State', 'uint32', 4, 4 + last_field_end_with_offset),
                    ('Pressure 11', 'float32', 4, 8 + last_field_end_with_offset),
                    ('Pressure 12', 'float32', 4, 12 + last_field_end_with_offset),
                    ('Integrated O3 Density 1', 'float32', 4, 16 + last_field_end_with_offset),
                    ('Pressure 21', 'float32', 4, 20 + last_field_end_with_offset),
                    ('Pressure 22', 'float32', 4, 24 + last_field_end_with_offset),
                    ('Integrated O3 Density 2', 'float32', 4, 28 + last_field_end_with_offset),
                    ('Pressure 31', 'float32', 4, 32 + last_field_end_with_offset),
                    ('Pressure 32', 'float32', 4, 36 + last_field_end_with_offset),
                    ('Integrated O3 Density 3', 'float32', 4, 40 + last_field_end_with_offset),
                    ('Pressure 41', 'float32', 4, 44 + last_field_end_with_offset),
                    ('Pressure 42', 'float32', 4, 48 + last_field_end_with_offset),
                    ('Integrated O3 Density 4', 'float32', 4, 52 + last_field_end_with_offset)]
        if product == "trg":
            fields = [
                    ('Selection Background State', 'uint32', 4, 4 + last_field_end_with_offset),
                    ('Integrated N20 Density', 'float32', 4, 8 + last_field_end_with_offset),
                    ('Integrated CO Density', 'float32', 4, 16 + last_field_end_with_offset),
                    ('Integrated CH4 Density', 'float32', 4, 20 + last_field_end_with_offset),
                    ('Integrated CO2 Density', 'float32', 4, 24 + last_field_end_with_offset)]
        if product == "ems":
            fields = []
        return fields


class Preprocessor:
    """
    A class used to handle the preprocessing of IASI data.

    This class is responsible for opening the binary files that contain the raw data,
    reading and structuring the data into a pandas DataFrame, and then doing a number
    of transformations on this data to prepare it for further analysis.

    Attributes:
    ----------
    intermediate_file : str
        The path to the binary file to be processed.
    data_level : str
        The level of the data to be processed ("l1c" or "l2").
    f : BinaryIO
        The binary file that is currently being processed.
    metadata : Metadata
        The metadata of the binary file that is currently being processed.
    data_record_df : pd.DataFrame
        The pandas DataFrame that stores the data extracted from the binary file.

    Methods:
    -------
    open_binary_file()
        Opens the binary file and extracts the metadata.
    close_binary_file()
        Closes the currently open binary file.
    flag_observations_to_keep(fields: List[Tuple])
        Creates a List of indices to sub-sample the main data set.
    read_record_fields(fields: List[Tuple])
        Reads the specified fields from the binary file and stores them in the DataFrame.
    read_spectral_radiance(fields: List[Tuple])
        Reads the spectral radiance data from the binary file and stores them in the DataFrame.
    build_local_time()
        Calculates and stores the local time at each point in the DataFrame.
    build_datetime()
        Combines the date and time fields into a single datetime field in the DataFrame.
    filter_bad_spectra(date: datetime)
        Filters out bad data based on IASI quality flags and overwrites the existing DataFrame.
    save_observations()
        Saves the observations in the DataFrame to a CSV file and deletes the intermediate binary file.
    preprocess_files(year: str, month: str, day: str)
        Runs the entire preprocessing pipeline on the binary file.
    """
    def __init__(self, ex: Extractor):
        self.intermediate_file: str = ex.intermediate_file
        self.data_level: str = ex.data_level
        self.latitude_range: Tuple[float] = ex.config.latitude_range
        self.longitude_range: Tuple[float] = ex.config.longitude_range
        self.channels: List[int] = ex.channels
        self.f: BinaryIO = None
        self.metadata: Metadata = None
        self.data_record_df = pd.DataFrame()

    @staticmethod
    def _get_common_fields() -> List[Tuple]:
        # Format of OBR fields List[(field_name, data_type)]
        fields = [
            ('Year', 'uint16'),
            ('Month', 'uint8'),
            ('Day', 'uint8'),
            ('Hour', 'uint8'),
            ('Minute', 'uint8'),
            ('Milliseconds', 'uint32'),
            ('Latitude', 'float32'),
            ('Longitude', 'float32'),
            ('Satellite Zenith Angle', 'float32'),
            ('Bearing', 'float32'),
            ('Solar Zenith Angle', 'float32'),
            ('Solar Azimuth', 'float32'),
            ('Field of View Number', 'uint32'),
            ('Orbit Number', 'uint32'),
            ('Scan Line Number', 'uint32'),
            ('Height of Station', 'float32')
            ]
        return fields
  
    @staticmethod
    def _get_l1c_record_fields() -> List[Tuple]:
        # Format of OBR fields List[(field_name, data_type)]
        fields = [
            ('Day version', 'uint16'),
            ('Start Channel 1', 'uint32'),
            ('End Channel 1', 'uint32'),
            ('Quality Flag 1', 'uint32'),
            ('Start Channel 2', 'uint32'),
            ('End Channel 2', 'uint32'),
            ('Quality Flag 2', 'uint32'),
            ('Start Channel 3', 'uint32'),
            ('End Channel 3', 'uint32'),
            ('Quality Flag 3', 'uint32'),
            ('Cloud Fraction', 'uint32'),
            ('Surface Type', 'uint8')
            ]
        return fields
    
    @staticmethod
    def _get_l1c_product_fields() -> List[Tuple]:
        # Format of OBR fields List[(field_name, data_type)]
        fields = [
            ('Spectrum', 'float32')
            ]
        return fields
      
    @staticmethod
    def _get_l2_record_fields() -> List[Tuple]:
        # Format of OBR fields List[(field_name, data_type)]
        fields = [
            ('Superadiabatic Indicator', 'uint8'),
            ('Land Sea Qualifier', 'uint8'),
            ('Day Night Qualifier', 'uint8'),
            ('Processing Technique', 'uint32'),
            ('Sun Glint Indicator', 'uint8'),
            ('Cloud Formation and Height Assignment', 'uint32'),
            ('Instrument Detecting Clouds', 'uint32'),
            ('Validation Flag for IASI L1 Product', 'uint32'),
            ('Quality Completeness of Retrieval', 'uint32'),
            ('Retrieval Choice Indicator', 'uint32'),
            ('Satellite Manoeuvre Indicator', 'uint32')
            ]
        return fields

    @staticmethod
    def _get_l2_product_fields() -> List[Tuple]:
        # Format of OBR fields List[(field_name, data_type)]
        fields = [
            ('Vertical Significance', 'uint32'),
            ('Pressure 1', 'float32'),
            ('Temperature or Dry Bulb Temperature 1', 'float32'),
            ('Cloud Amount in Segment 1', 'float32'),
            ('Cloud Phase 1', 'uint32'),
            ('Pressure 2', 'float32'),
            ('Temperature or Dry Bulb Temperature 2', 'float32'),
            ('Cloud Amount in Segment 2', 'float32'),
            ('Cloud Phase 2', 'uint32'),
            ('Pressure 3', 'float32'),
            ('Temperature or Dry Bulb Temperature 3', 'float32'),
            ('Cloud Amount in Segment 3', 'float32'),
            ('Cloud Phase 3', 'uint32')
            ]
        return fields

    @staticmethod
    def process_chunk(chunk: pd.DataFrame, dtype_dict: dict) -> pd.DataFrame:
        # Find the intersection of chunk columns and dtype_dict keys, and convert it to a list
        relevant_cols = list(set(chunk.columns) & set(dtype_dict.keys()))
        
        # Apply type conversion in a vectorized manner
        chunk[relevant_cols] = chunk[relevant_cols].astype({col: dtype_dict[col] for col in relevant_cols})
        
        # Find columns in chunk but not in dtype_dict and print them
        missing_cols = list(set(chunk.columns) - set(dtype_dict.keys()))
        if missing_cols:
            print("Columns in DataFrame but not in dtype_dict:", missing_cols)
            
        return chunk
 
    def open_text_file(self) -> None:
        print("\nLoading intermediate text file:")    
        
        # Read and combine byte tables to optimise reading of OBR txtfile
        combined_fields = (Preprocessor._get_common_fields() +
                           Preprocessor._get_l1c_record_fields() +
                           Preprocessor._get_l1c_product_fields() +
                           Preprocessor._get_l2_record_fields() +
                           Preprocessor._get_l2_product_fields()
                           )

        # Create dtype dict from combined fields
        dtype_dict = {field[0]: field[1] for field in combined_fields}
        
        # Initialise an empty DataFrame to hold the processed chunks
        processed_data = pd.DataFrame()

        # Specify the chunk size
        chunk_size = 1000
        # Iterate over the CSV file in chunks
        for i, chunk in enumerate(pd.read_csv(self.intermediate_file, sep="\t", dtype=dtype_dict, chunksize=chunk_size)):
            # Process each chunk using the static method
            processed_chunk = Preprocessor.process_chunk(chunk, dtype_dict)
            
            # Append the processed chunk to the DataFrame
            processed_data = pd.concat([processed_data, processed_chunk], ignore_index=True)

        # Assign the concatenated processed data back to self.data_record_df
        self.data_record_df = processed_data
        print(self.data_record_df.info(verbose=True))
        input()
        return


    def fix_spectrum_columns(self) -> None:
        # Rename columns based on the integer list of channel IDs
        rename_mapping = {str(self.channels[0] + i): f"Spectrum {channel_id}" for i, channel_id in enumerate(self.channels)}
        self.data_record_df.rename(columns=rename_mapping, inplace=True)
        return


    def _calculate_local_time(self) -> None:
        """
        Calculate the local time (in hours, UTC) that determines whether it is day or night at a specific longitude.

        Returns:
        np.ndarray: Local time (in hours, UTC) within a 24 hour range, used to determine day (6-18) or night (0-6, 18-23).
        """

        # Retrieve the necessary field data
        hour, minute, millisecond, longitude = self.data_record_df['Hour'], self.data_record_df['Minute'], self.data_record_df['Milliseconds'], self.data_record_df['Longitude']

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


    def build_local_time(self) -> List:
        """
        Stores the local time Boolean indicating whether the current time is day or night.
        """
        print("\nBuilding Local Time:")
        # Calculate the local time
        local_time = self._calculate_local_time()

        # Store the Boolean indicating day (True) or night (False) in the DataFrame
        self.data_record_df['Local Time'] = (6 < local_time) & (local_time < 18)
        return


    def build_datetime(self) -> List:
        """
        Stores the datetime components to a single column and drops the elements.
        """
        print("\nBuilding Datetime:")
        self.data_record_df['Datetime'] = (self.data_record_df['Year'].apply(lambda x: f'{int(x):04d}') +
                                    self.data_record_df['Month'].apply(lambda x: f'{int(x):02d}') +
                                    self.data_record_df['Day'].apply(lambda x: f'{int(x):02d}') +
                                    self.data_record_df['Hour'].apply(lambda x: f'{int(x):02d}') +
                                    self.data_record_df['Minute'].apply(lambda x: f'{int(x):02d}')
                                    )
        
        # Drop original time element columns
        self.data_record_df = self.data_record_df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Milliseconds'])
        return  
    

    def _delete_intermediate_file(self) -> None:
        os.remove(self.intermediate_file)
        return

    def save_observations(self, delete_obr_file: bool = True) -> None:
        """
        Saves the observation data to CSV/HDF5 file and deletes OBR output file.
        """  
        # Create output file name
        outfile = self.intermediate_file.split(".")[0]
        print(f"\nSaving DataFrame to: {outfile}.pkl.gz")

        # Compress and save using gzip
        with gzip.open(outfile, 'wb') as f:
            pickle.dump(self.data_record_df, f)
        
        # Delete intermediate OBR output file
        if delete_obr_file == True:
            self._delete_intermediate_file()
        return