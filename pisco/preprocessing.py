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
    
    def _print_metadata(self) -> None:
        print(f"Header  : {self.header_size} bytes")
        print(f"Record  : {self.record_size} bytes")
        print(f"Data    : {self.number_of_measurements} measurements")
        return

    def _verify_header(self) -> None:
        """
        Verifies the header size by comparing it with the header size at the end of the header.
        """
        # Reset file pointer to the beginning
        self.f.seek(0)

        # Skip first int32 value
        np.fromfile(self.f, dtype='uint32', count=1)

        # Skip header content
        np.fromfile(self.f, dtype='uint8', count=self.header_size)

        # Read header size at the end of the header
        header_size_check = np.fromfile(self.f, dtype='uint32', count=1)[0]

        # Check if header sizes match
        assert self.header_size == header_size_check, "Header size mismatch"

    def _count_measurements(self) -> int:
        """
        Calculate the number of measurements in the binary file based on its size, 
        header size and record size.

        Returns:
        int: The number of measurements.
        """
        # Get the total size of the file
        file_size = self.f.seek(0, 2)
        # Calculate the number of measurements (minus 1 to avoid erroneous reads at the end of the byte structure)
        self.number_of_measurements = int(((file_size - self.header_size - 8) // (self.record_size + 8)) - 1)
        return
    
    def _read_header_record_size(self, common_header_fields: List[Tuple]) -> None:
        # Read header size
        _, dtype, dtype_size, cumsize = self._get_field_from_tuples('Header Size', common_header_fields)
        self.f.seek(cumsize-dtype_size, 0)
        self.header_size = int(np.fromfile(self.f, dtype=dtype, count=1)[0])

        # Read record size
        _, dtype, dtype_size, cumsize = self._get_field_from_tuples('Record Header Size', common_header_fields)
        self.f.seek(cumsize-dtype_size, 0)
        self.record_size = int(np.fromfile(self.f, dtype=dtype, count=1)[0])
        return


    def _get_field_from_tuples(self, key, tuples_list) -> Optional[Tuple]:
        try:
            for tup in tuples_list:
                if tup[0] == key:
                    return tup
            raise ValueError(f"Key '{key}' not found in tuples list")
        except Exception as e:
            print(f"Error in _get_value_from_tuples: {e}")
            raise


    def _get_fixed_size_fields_pre(self) -> List[Tuple]:
        "Byte table for values occuring before Channel IDs"
        pre_channel_id_fields = [
            ('Header Size', 'uint32', 4, 4),
            ('Byte Order', 'uint8', 1, 5),
            ('Format Version', 'uint32', 4, 9),
            ('Satellite Identifier', 'uint32', 4, 13),
            ('Record Header Size', 'uint32', 4, 17),
            ('Brightness Temperature Brilliance', 'bool', 1, 18),
            ('Number of Channels', 'uint32', 4, 22)
        ]
        return pre_channel_id_fields

    def _get_fixed_size_fields_post(self, channel_id_field: Tuple) -> List[Tuple]:
        "Byte table for values occuring after Channel IDs"
        # Get the tuple for 'Channel IDs'
        field, dtype, dtype_size, cumsize = self._get_field_from_tuples('Channel IDs', channel_id_field)
        post_channel_id_fields = [
            ('AVHRR Brilliance', 'bool', 1, 1 + cumsize),
            ('Number of L2 Products', 'uint16', 2, 3 + cumsize)
        ]
        return post_channel_id_fields
    

    def _read_channel_ids(self, cumsize: int) -> None:
        self.f.seek(cumsize, 0)
        self.channel_IDs = np.fromfile(self.f, dtype='uint32', count=self.number_of_channels)
        return
    
    def _get_channel_id_field(self, pre_channel_id_fields: List[Tuple]):
        """This is variable and treated separately."""
        # Get the tuple for 'Number of Channels'
        _, dtype, dtype_size, cumsize = self._get_field_from_tuples('Number of Channels', pre_channel_id_fields)
        self.f.seek(cumsize-dtype_size, 0)
        self.number_of_channels = int(np.fromfile(self.f, dtype=dtype, count=1)[0])

        # Store for later
        self._read_channel_ids(cumsize)

        # Return as a list for concatenation with other fields
        return [('Channel IDs', 'uint32', 4 * self.number_of_channels, cumsize + (4 * self.number_of_channels))]

    
    def _read_l2_product_ids(self, cumsize: int) -> None:
        self.f.seek(cumsize, 0)
        self.l2_product_IDs = np.fromfile(self.f, dtype='uint32', count=self.number_of_l2_products)
        return
        
    def _get_l2_product_id_field(self, post_channel_id_fields: List[Tuple]):
        """This is variable and treated separately."""
        # Get the tuple for 'Number of L2 Products'
        _, dtype, dtype_size, cumsize = self._get_field_from_tuples('Number of L2 Products', post_channel_id_fields)
        self.f.seek(cumsize-dtype_size, 0)
        self.number_of_l2_products = int(np.fromfile(self.f, dtype=dtype, count=1)[0])
        
        # Store for later
        self._read_l2_product_ids(cumsize)

        # Return as a list for concatenation with other fields
        return [('L2 Product IDs', 'uint32', 4 * self.number_of_l2_products, cumsize + (4 * self.number_of_l2_products))]
    

    def _build_iasi_common_header_fields(self) -> List[Tuple]:
        # Step 1: Get pre-channel ID fields
        pre_channel_id_fields = self._get_fixed_size_fields_pre()
        
        # Step 2: Get channel ID field
        channel_id_field = self._get_channel_id_field(pre_channel_id_fields)

        # Step 3: Get post-channel ID fields
        post_channel_id_fields = self._get_fixed_size_fields_post(channel_id_field)

        # Step 4: Get L2 product ID field
        l2_product_id_field = self._get_l2_product_id_field(post_channel_id_fields)
        return pre_channel_id_fields + channel_id_field + post_channel_id_fields + l2_product_id_field
    
    def check_iasi_common_header(self) -> None:
        common_header_fields = self._build_iasi_common_header_fields()
        self._read_header_record_size(common_header_fields)
        self._count_measurements()
        self._print_metadata()
        return

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
    def _get_end_of_common_record():
        return Metadata._get_iasi_common_record_fields()[-1][-1]  # End of the Height of Station field
    

    @staticmethod
    def _get_iasi_l1c_record_fields() -> List[Tuple]:
        offset = Metadata._get_end_of_common_record()

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
    def _get_end_of_l1c_record():
        # Determine the position of the anchor point for spectral radiance data in the binary file
        return Metadata._get_iasi_l1c_record_fields()[-1][-1]  # End of the Surface Type field

    # def _get_l1c_product_record_fields(self) -> List[Tuple]:
    #     offset = Metadata._get_end_of_l1c_record()
        
    #     # Format of L1Cspectral radiance fields in binary file (field_name, data_type, data_size, cumulative_data_size)
    #     fields = [('Spectrum', 'float32', 4 * self.number_of_channels, (4 * self.number_of_channels) + offset)]
    #     return fields
    

    @staticmethod
    def _get_iasi_l2_record_fields() -> List[Tuple]:
        offset = Metadata._get_end_of_common_record()

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
    def _get_end_of_l2_record(product_index: int, product_ID: int):
        # Determine the position of the anchor point for spectral radiance data in the binary file
        last_field_end =  Metadata._get_iasi_l2_record_fields()[-1][-1]  # End of the Satellite Manoeuvre Indicator field
        # Shift cumsizes by offset equal to number of other L2 products already read
        last_field_end_with_offset = last_field_end * (product_index + 1)
        
        # Use product ID to extract relevant L2 product
        l2_product_dictionary = {1: "clp", 2: "twt", 3: "ozo", 4: "trg", 5: "ems"}
        product = l2_product_dictionary.get(product_ID)
        return last_field_end_with_offset, product

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
    def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        # Select all int64 columns and convert them to int32
        int_cols = chunk.select_dtypes(include='int64').columns
        chunk[int_cols] = chunk[int_cols].astype('int32')

        # Select all float64 columns and convert them to float32
        float_cols = chunk.select_dtypes(include='float64').columns
        chunk[float_cols] = chunk[float_cols].astype('float32')

        return chunk
    
    def open_text_file(self) -> None:
        print("\nLoading intermediate text file:")    
        
        # Read and combine byte tables to optimise reading of OBR txtfile
        combined_fields = (Metadata._get_iasi_common_record_fields() +
                           Metadata._get_iasi_l1c_record_fields() +
                           [('Spectrum', 'float32')] +
                           Metadata._get_iasi_l2_record_fields() +
                           Metadata._get_l2_product_fields('clp')
                           )
        
        print(combined_fields)

        exit()

        # Create dtype dict from combined fields
        dtype_dict = {field[0]: field[1] for field in combined_fields}

        
        # Initialise an empty DataFrame to hold the processed chunks
        processed_data = pd.DataFrame()

        # Specify the chunk size
        chunk_size = 1000
        # Iterate over the CSV file in chunks
        for i, chunk in enumerate(pd.read_csv(self.intermediate_file, sep="\t", chunksize=chunk_size)):
            # Process each chunk using the static method
            processed_chunk = Preprocessor.process_chunk(chunk)
            
            # Append the processed chunk to the DataFrame
            processed_data = pd.concat([processed_data, processed_chunk], ignore_index=True)

        # Assign the concatenated processed data back to self.data_record_df
        self.data_record_df = processed_data
        print(self.data_record_df.info(verbose=True))
        return
    
    def fix_spectrum_columns(self) -> None:
        # Rename columns based on the integer list of channel IDs
        rename_mapping = {str(self.channels[0] + i): f"Spectrum {channel_id}" for i, channel_id in enumerate(self.channels)}
        self.data_record_df.rename(columns=rename_mapping, inplace=True)
        return

    def open_binary_file(self) -> None:
        print("\nLoading intermediate binary file:")
        self.f = open(self.intermediate_file, 'rb')
        
        # Get structure of file header and data record
        self.metadata = Metadata(self.f)
        self.metadata.check_iasi_common_header()
        return

    def close_binary_file(self):
        self.f.close()
        return       


    def _calculate_byte_offset(self, dtype_size: int) -> int:
        return self.metadata.record_size + 8 - dtype_size
    
    def _set_field_start_position(self, cumsize: int) -> None:
        self.f.seek(self.metadata.header_size + 12 + cumsize, 0)
        return
    
    def _store_data_in_df(self, field: str, data: np.ndarray) -> None:
        if not field == "Spectrum":
            self.data_record_df[field] = data
        else:
            # Prepare new columns for the spectrum data
            spectrum_columns = {f'Spectrum {channel_ID}': data[i, :] for i, channel_ID in enumerate(self.metadata.channel_IDs)}
            
            # Create a new DataFrame from the spectrum columns
            spectrum_df = pd.DataFrame(spectrum_columns)

            # Concatenate this new DataFrame with the existing one
            self.data_record_df = pd.concat([self.data_record_df, spectrum_df], axis=1)
        return

    def _read_binary_data(self, field: str, dtype: Any, dtype_size: int) -> np.ndarray:
        """
        Reads the data of each measurement based on the valid indices.

        Args:
            dtype (Any): Data type of the field.
            dtype_size (int): Data type size in Bytes.

        Returns:
            np.ndarray: 1-D array of field data.
        """
        # Calculate the byte offset to the next measurement
        byte_offset = self._calculate_byte_offset(dtype_size)
        
        # Calculate byte location to start pointer (skipping invalid indices)
        byte_start = (byte_offset + dtype_size)
        # Move file pointer to first valid index
        self.f.seek(byte_start, 1)

        # Iterate over field elements and extract values from binary file.
        # Split conditions to avoid evaluting if statements at each iteration.
        if not field == "Spectrum":
            # Prepare an NaN array to store the data of the current field
            data = np.full(self.metadata.number_of_measurements, np.nan, dtype="float32")
            for i in range(self.metadata.number_of_measurements):
                
                # Read the field for the current measurement
                step = (byte_offset * i) + (dtype_size * (i - 1))
                value = np.fromfile(self.f, dtype=dtype, count=1, sep='', offset=step)

                # Store the value in the data array if value exists; leave untouched otherwise (as np.nan).
                data[i] = value[0] if len(value) != 0 else data[i]
        else:
            # Prepare an NaN array to store the data of the spectrum field
            data = np.full((self.metadata.number_of_channels, self.metadata.number_of_measurements), np.nan, dtype="float32")
            for i in range(self.metadata.number_of_measurements):
                
                # Read the value for the current measurement
                step = (byte_offset * i) + (dtype_size * (i - 1))
                
                # Store the value in the data array if value exists; leave untouched otherwise (as np.nan).
                spectrum = np.fromfile(self.f, dtype='float32', count=self.metadata.number_of_channels, sep='', offset=step)
                data[:, i] = spectrum if len(spectrum) != 0 else data[:, i]

        # Store the data in the DataFrame
        self._store_data_in_df(field, data)
        return
          
    def read_record_fields(self, fields: List[Tuple]) -> None:
        """
        Reads the data of each field from the binary file and stores it in a pandas DataFrame.

        Args:
            fields (List[Tuple]): List of field tuples containing field information.

        Returns:
            None
        """        
        for field, dtype, dtype_size, cumsize in fields:
            # Print field extraction progress
            print(f"Extracting: {field}", dtype, dtype_size, cumsize)
            
            # Set the file pointer to the start position of the field
            self._set_field_start_position(cumsize)
            
            # Read the binary data based on the valid indices
            self._read_binary_data(field, dtype, dtype_size)
    
    def read_l2_product_fields(self):
        # Retrieve the individual L2 products from the configuration file
        for product_index, product_ID in enumerate(self.metadata.l2_product_IDs):
            offset, product = Metadata._get_end_of_l2_record(product_index, product_ID)
            self.read_record_fields(Metadata._get_l2_product_fields(product))


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

        # Save the DataFrame to a file in HDF5 format
        # # self.data_record_df.to_hdf(f"{datapath_out}{datafile_out}.h5", key='df', mode='w')
        # self.data_record_df.to_csv(f"{outfile}.csv", index=False, mode='w')

        # Compress and save using gzip
        with gzip.open(outfile, 'wb') as f:
            pickle.dump(self.data_record_df, f)
        
        # Delete intermediate OBR output file
        if delete_obr_file == True:
            self._delete_intermediate_file()
        return