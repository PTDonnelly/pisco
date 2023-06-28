from datetime import datetime
import os
import numpy as np
import pandas as pd
from typing import Any, List, BinaryIO, Tuple, Set

import numpy as np

class Metadata:
    """
    Metadata class provides the structure and methods to read and process
    metadata of binary files.

    Attributes:
        f (BinaryIO): The binary file object.
        header_size (int): The size of the header in bytes.
        byte_order (int): The byte order of the binary data.
        format_version (int): The version of the data format.
        satellite_identifier (int): The identifier of the satellite.
        record_header_size (int): The size of the record header.
        brightness_temperature_brilliance (bool): The brightness temperature brilliance.
        number_of_channels (int): The number of channels.
        channel_IDs (np.array): The IDs of the channels.
        AVHRR_brilliance (bool): The AVHRR brilliance.
        number_of_L2_sections (int): The number of Level 2 sections.
        record_size (int): The size of each record in bytes.
        number_of_measurements (int): The number of measurements.
    """
    def __init__(self, file: BinaryIO):
        self.f: BinaryIO = file
        self.header_size: int = None
        self.byte_order: int = None
        self.format_version: int = None
        self.satellite_identifier: int = None
        self.record_header_size: int = None
        self.brightness_temperature_brilliance: bool = None
        self.number_of_channels: int = None
        self.channel_IDs: np.array = None
        self.AVHRR_brilliance: bool = None
        self.number_of_L2_sections: int = None
        self.table_of_L2_sections: int = None
        self.record_size: int = None
        self.number_of_measurements: int = None
    
    def _print_metadata(self) -> None:
        print(f"Header  : {self.header_size} bytes")
        print(f"Record  : {self.record_size} bytes")
        print(f"Spectrum: {self.number_of_channels} channels")
        print(f"Data    : {self.number_of_measurements} measurements")
        return

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
        self.number_of_measurements = ((file_size - self.header_size - 8) // (self.record_size + 8)) - 1
        return
    
    def _read_record_size(self) -> int:
        self.f.seek(self.header_size + 8, 0)
        self.record_size = np.fromfile(self.f, dtype='uint32', count=1)[0]
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

    def _read_iasi_common_header_metadata(self) -> None:
        """
        Reads the header of the binary file to obtain the header size and number of channels.
        """
        # Read header entries
        self.header_size = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.byte_order = np.fromfile(self.f, dtype='uint8', count=1)[0]
        self.format_version = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.satellite_identifier = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.record_header_size = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.brightness_temperature_brilliance = np.fromfile(self.f, dtype='bool', count=1)[0]
        self.number_of_channels = np.fromfile(self.f, dtype='uint32', count=1)[0]
        self.channel_IDs = np.fromfile(self.f, dtype='uint32', count=self.number_of_channels)
        self.AVHRR_brilliance = np.fromfile(self.f, dtype='bool', count=1)[0]
        self.number_of_L2_sections = np.fromfile(self.f, dtype='uint16', count=1)[0]
        # self.table_of_L2_sections = np.fromfile(self.f, dtype='uint32', count=self.number_of_L2_sections)[0]

        # Read header size at the end of the header, check for a match
        self._verify_header()       
        return
    
    def get_iasi_common_header(self) -> None:
        self._read_iasi_common_header_metadata()
        self._read_record_size()
        self._count_measurements()
        self._print_metadata()
        return


    def _get_iasi_common_record_fields(self) -> List[tuple]:
        # Format of fields in binary file (field_name, data_type, data_size, cumulative_data_size)
        common_fields = [
                        ('Year', 'uint16', 2, 2),
                        ('Month', 'uint8', 1, 3),
                        ('Day', 'uint8', 1, 4),
                        ('Hour', 'uint8', 1, 5),
                        ('Minute', 'uint8', 1, 6),
                        ('Millisecond', 'uint32', 4, 10),
                        ('Latitude', 'float32', 4, 14),
                        ('Longitude', 'float32', 4, 18),
                        ('Satellite Zenith Angle', 'float32', 4, 22),
                        ('Bearing', 'float32', 4, 26),
                        ('Solar Zentih Angle', 'float32', 4, 30),
                        ('Solar Azimuth', 'float32', 4, 34),
                        ('Field of View Number', 'uint32', 4, 38),
                        ('Orbit Number', 'uint32', 4, 42),
                        ('Scan Line Number', 'uint32', 4, 46),
                        ('Height of Station', 'float32', 4, 50)]
        return common_fields
    
    def _get_iasi_l1c_record_fields(self) -> List[tuple]:
        # Format of L1C-specific fields in binary file (field_name, data_type, data_size, cumulative_data_size),
        # cumulative total continues from the fourth digit of the last tuple in common_fields.
        l1c_fields = [
                    ('Day version', 'uint16', 2, 52),
                    ('Start Channel 1', 'uint32', 4, 56),
                    ('End Channel 1', 'uint32', 4, 60),
                    ('Quality Flag 1', 'uint32', 4, 64),
                    ('Start Channel 2', 'uint32', 4, 68),
                    ('End Channel 2', 'uint32', 4, 72),
                    ('Quality Flag 2', 'uint32', 4, 76),
                    ('Start Channel 3', 'uint32', 4, 80),
                    ('End Channel 3', 'uint32', 4, 84),
                    ('Quality Flag 3', 'uint32', 4, 88),
                    ('Cloud Fraction', 'uint32', 4, 92),
                    ('Surface Type', 'uint8', 1, 93)]
        return l1c_fields
    
    def _get_iasi_l2_record_fields(self) -> List[tuple]:
        # Format of L2-specific fields in binary file (field_name, data_type, data_size, cumulative_data_size),
        # cumulative total continues from the fourth digit of the last tuple in common_fields.
        l2_fields = [
                    ('Superadiabatic Indicator', 'uint8', 1, 51),
                    ('Land Sea Qualifier', 'uint8', 1, 52),
                    ('Day Night Qualifier', 'uint8', 1, 53),
                    ('Processing Technique', 'uint32', 4, 57),
                    ('Sun Glint Indicator', 'uint8', 1, 58),
                    ('Cloud Formation and Height Assignment', 'uint32', 4, 62),
                    ('Instrument Detecting Clouds', 'uint32', 4, 66),
                    ('Validation Flag for IASI L1 Product', 'uint32', 4, 70),
                    ('Quality Completeness of Retrieval', 'uint32', 4, 74),
                    ('Retrieval Choice Indicator', 'uint32', 4, 78),
                    ('Satellite Manoeuvre Indicator', 'uint32', 4, 82)]
        return l2_fields
    
    def _get_ozo_record_fields(self):
        pass

    def _get_trg_record_fields(self):
        pass
    
    def _get_clp_record_fields(self):
        pass

    def _get_twt_record_fields(self):
        pass

    def _get_ems_record_fields(self):
        pass


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
    flag_observations_to_keep(fields: List[tuple])
        Creates a Set of indices to sub-sample the main data set.
    read_record_fields(fields: List[tuple])
        Reads the specified fields from the binary file and stores them in the DataFrame.
    read_spectral_radiance(fields: List[tuple])
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
    def __init__(self, intermediate_file: str, data_level: str, latitude_range: Tuple[float], longitude_range: Tuple[float]):
        self.intermediate_file = intermediate_file
        self.data_level = data_level
        self.latitude_range = latitude_range
        self.longitude_range = longitude_range
        self.f: BinaryIO = None
        self.metadata: Metadata = None
        self.data_record_df = pd.DataFrame()


    def open_binary_file(self) -> None:
        # Open binary file
        print("\nLoading intermediate L1C file...")
        self.f = open(self.intermediate_file, 'rb')
        
        # Get structure of file header and data record
        self.metadata = Metadata(self.f)
        self.metadata.get_iasi_common_header()
        return

    def close_binary_file(self):
        self.f.close()
        return
    

    def _get_indices(self, field: str, dtype: Any, byte_offset: int) -> Set[int]:
        """
        Read and check the indices of measurements based on the latitude or longitude values.

        Args:
            field (str): Field name, either 'Latitude' or 'Longitude'.
            dtype (Any): Data type of the field.
            byte_offset (int): Byte offset to the next measurement.

        Returns:
            Set[int]: Set of indices of measurements that fall within the specified range.
        """
        # Initialize an array to store measurement values
        values = np.empty(self.metadata.number_of_measurements)
        
        # Define an empty set to hold valid indices
        valid_indices = set()

        # Loop through each measurement in the data
        for measurement in range(self.metadata.number_of_measurements):
            # Read and store the value of the field from the file
            values[measurement] = np.fromfile(self.f, dtype=dtype, count=1, sep='', offset=byte_offset)
            # progress = np.round((measurement / self.metadata.number_of_measurements) * 100, 2)
            # if progress % 1 == 0:
            #     print(f"{measurement} / {self.metadata.number_of_measurements} = {progress}")

        # Given the field, filter the indices based on the specified range
        if field == 'Latitude':
            valid_indices = set(np.where((self.latitude_range[0] <= values) & (values <= self.latitude_range[1]))[0])
        # If the field is 'Longitude', filter the indices based on the longitude range
        elif field == 'Longitude':
            valid_indices = set(np.where((self.longitude_range[0] <= values) & (values <= self.longitude_range[1]))[0])

        # Return the indices that fall within the specified range for the given field
        return valid_indices

    def _calculate_byte_offset(self, dtype_size: int) -> int:
        return self.metadata.record_size + 8 - dtype_size
    
    def _set_field_start_position(self, cumsize: int) -> None:
        self.f.seek(self.metadata.header_size + 12 + cumsize, 0)
        return
    
    def _check_spatial_range(self):
        return True if (self.latitude_range == [-90, 90]) & (self.longitude_range == [-180, 180]) else False
    
    def flag_observations_to_keep(self, fields: List[tuple]) -> Set[int]:
        """
        Go through the latitude and longitude fields to find and store indices of measurements 
        where latitude and longitude fall inside the specified range.

        If the latitude and longitude cover the full globe, return a set of all indices.

        If the latitude and longitude do not cover the full globe, find the valid indices 
        based on the specified range and return the intersection of valid latitude and longitude indices.

        Args:
            fields (List[tuple]): List of field tuples containing field information.

        Returns:
            Set[int]: Set of indices of measurements to be processed in the main loop.
        """
        # Check if the latitude and longitude cover the full globe
        full_globe = self._check_spatial_range()
        
        if full_globe:
            # If the latitude and longitude cover the full globe, return all indices
            return set(range(self.metadata.number_of_measurements))
        else:
            print(f"\nFlagging observations to keep...")

            for field, dtype, dtype_size, cumsize in fields:
                if field not in ['Latitude', 'Longitude']:
                    # Skip all other fields for now
                    continue

                # Set the starting position of the field and calculate the byte offset
                self._set_field_start_position(cumsize)
                byte_offset = self._calculate_byte_offset(dtype_size)

                # Read and store the valid indices for the field
                valid_indices = self._get_indices(field, dtype, byte_offset)
                if field == 'Latitude':
                    valid_indices_lat = valid_indices
                elif field == 'Longitude':
                    valid_indices_lon = valid_indices

            # Return the intersection of valid latitude and longitude indices
            return valid_indices_lat & valid_indices_lon
            

    def _store_data_in_df(self, field: str, data: np.ndarray) -> None:
        self.data_record_df[field] = data
        return

    def _read_binary_data(self, valid_indices: Set[int], dtype: Any, byte_offset: int) -> np.ndarray:
        """
        Reads the data of each measurement based on the valid indices.

        Args:
            valid_indices (Set[int]): Set of valid measurement indices.
            dtype (Any): Data type of the field.
            byte_offset (int): Byte offset to the next measurement.

        Returns:
            np.ndarray: Array of field data.
        """
        # Prepare an empty array to store the data of the current field
        data = np.empty(len(valid_indices))

        # #### TEST THIS, IT COULD IMPROVE THE SPEED A FAIR BIT
        # for i, measurement in enumerate(valid_indices):
        #     # Move file pointer to value
        #     self.f.seek(byte_offset * measurement, 1)
        #     # Read the value for the current measurement
        #     value = np.fromfile(self.f, dtype=dtype, count=1, sep='')
        #     # Store the value in the data array, handling missing values as NaN
        #     data[i] = np.nan if len(value) == 0 else value[0]
        # #####
        
        
        # Counter for the valid indices in data
        valid_index = 0
        
        for measurement in range(self.metadata.number_of_measurements):
            # Read the value for the current measurement
            value = np.fromfile(self.f, dtype=dtype, count=1, sep='', offset=byte_offset)
            if measurement in valid_indices:
                print(measurement, byte_offset, byte_offset * measurement, self.f.tell())
                input()
                # Store the value in the data array, handling missing values as NaN
                data[valid_index] = np.nan if len(value) == 0 else value[0]
                # Increment the valid index counter
                valid_index += 1
            else:
                # Skip this measurement
                continue

        return data
          
    def read_record_fields(self, fields: List[tuple], valid_indices: Set[int]) -> None:
        """
        Reads the data of each field from the binary file and stores it in a pandas DataFrame.

        Args:
            fields (List[tuple]): List of field tuples containing field information.
            valid_indices (Set[int]): Set of valid indices to process.

        Returns:
            None
        """        
        for field, dtype, dtype_size, cumsize in fields:
            # Print field extraction progress
            print(f"Extracting: {field}")

            # Set the file pointer to the start position of the field
            self._set_field_start_position(cumsize)

            # Calculate the byte offset to the next measurement
            byte_offset = self._calculate_byte_offset(dtype_size)

            # Read the binary data based on the valid indices
            data = self._read_binary_data(valid_indices, dtype, byte_offset)

            # Store the data in the DataFrame
            self._store_data_in_df(field, data)

    
    def _store_spectral_channels_in_df(self, data: np.ndarray) -> None:
        for i, channel_id in enumerate(self.metadata.channel_IDs):
            self.data_record_df[f'Spectrum {channel_id}'] = data[i, :]
        return
    
    def _read_spectrum(self, valid_indices: Set[int], byte_offset: int) -> np.ndarray:
        """
        Read the spectral radiance data for valid measurements.

        Args:
            valid_indices (Set[int]): Set of valid measurement indices.
            byte_offset (int): Byte offset to the next measurement.

        Returns:
            np.ndarray: Array of spectral radiance data.
        """
        # Initialize an empty numpy array to store the spectral radiance data
        data = np.empty((self.metadata.number_of_channels, len(valid_indices)))

    #    #### TEST THIS, IT COULD IMPROVE THE SPEED A FAIR BIT
    #     for i, measurement in enumerate(valid_indices):
    #         # Move file pointer to value
    #         self.f.seek(byte_offset * measurement, 1)
    #         # Read the value for the current measurement
    #         spectrum = np.fromfile(self.f, dtype="float32", count=self.metadata.number_of_channels, sep='')
    #         # Store the value in the data array, handling missing values as NaN
    #         data[:, i] = np.nan if len(spectrum) == 0 else spectrum
    #     #####
        
        # Counter for the valid indices in data
        valid_index = 0

        # Iterate over each measurement and extract the spectral radiance data
        for measurement in range(self.metadata.number_of_measurements):
            # Read the spectrum data for the valid measurement
            spectrum = np.fromfile(self.f, dtype='float32', count=self.metadata.number_of_channels, sep='', offset=byte_offset)
            if measurement in valid_indices:
                # Store the spectrum in the data array, handling missing values as NaN
                data[:, valid_index] = np.nan if len(spectrum) == 0 else spectrum
                # Increment the valid index counter
                valid_index += 1
            else:
                # Skip this measurement
                continue

        return data

    def _calculate_byte_offset_spectral_radiance(self) -> int:
        return self.metadata.record_size + 8 - (4 * self.metadata.number_of_channels)
    
    def _set_start_read_position(self, last_field_end: int) -> None:
        self.f.seek(self.metadata.header_size + 12 + last_field_end + (4 * self.metadata.number_of_channels), 0)
        return
    
    def read_spectral_radiance(self, fields: List[tuple], valid_indices: Set[int]) -> None:
        """
        Extracts and stores the spectral radiance measurements from the binary file.

        Args:
            fields (List[tuple]): List of field tuples containing field information.
            valid_indices (Set[int]): Set of valid measurement indices to process.

        Returns:
            None
        """
        print("Extracting: radiance")

        # Determine the position of the anchor point for spectral radiance data in the binary file
        last_field_end = fields[-1][-1]  # End of the surface_type field

        # Set the start read position based on the last field end
        self._set_start_read_position(last_field_end)

        # Calculate the byte offset to skip to the next measurement for spectral radiance data
        byte_offset = self._calculate_byte_offset_spectral_radiance()

        # Read the spectral radiance data for the valid indices
        data = self._read_spectrum(valid_indices, byte_offset)

        # Store the spectral channels in the DataFrame
        self._store_spectral_channels_in_df(data)


    def _calculate_local_time(self) -> None:
        """
        Calculate the local time (in hours, UTC) that determines whether it is day or night at a specific longitude.

        Returns:
        np.ndarray: Local time (in hours, UTC) within a 24 hour range, used to determine day (6-18) or night (0-6, 18-23).
        """

        # Retrieve the necessary field data
        hour, minute, millisecond, longitude = self.data_record_df['Hour'], self.data_record_df['Minute'], self.data_record_df['Millisecond'], self.data_record_df['Longitude']

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
        print("\nBuilding Local Time...")
        # Calculate the local time
        local_time = self._calculate_local_time()

        # Store the Boolean indicating day (True) or night (False) in the DataFrame
        self.data_record_df['Local Time'] = (6 < local_time) & (local_time < 18)
        return


    def build_datetime(self) -> List:
        """
        Stores the datetime components to a single column and drops the elements.
        """
        print("\nBuilding Datetime...")
        # Create 'Datetime' column
        self.data_record_df['Datetime'] = self.data_record_df['Year'].apply(lambda x: f'{int(x):04d}') + \
                                    self.data_record_df['Month'].apply(lambda x: f'{int(x):02d}') + \
                                    self.data_record_df['Day'].apply(lambda x: f'{int(x):02d}') + '.' + \
                                    self.data_record_df['Hour'].apply(lambda x: f'{int(x):02d}') + \
                                    self.data_record_df['Minute'].apply(lambda x: f'{int(x):02d}') + \
                                    self.data_record_df['Millisecond'].apply(lambda x: f'{int(x/10000):02d}')

        # Drop original time element columns
        self.data_record_df = self.data_record_df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Millisecond'])
        return  
    

    def filter_good_spectra(self, date: object) -> None:
        """
        Filters bad spectra based on IASI L1C data quality flags and date. Overwrites existing DataFrame.
        """
        print("\nFiltering spectra...")
        if date <= datetime(2012, 2, 8):
            # Treat data differently if before February 8 2012 (due to a change in IASI data reduction)
            check_quality_flag = self.data_record_df['Quality Flag'] == 0
            check_data = self.data_record_df.drop(['Quality Flag', 'Date Column'], axis=1).sum(axis=1) > 0
            good_flag = check_quality_flag & check_data
        else:
            check_quality_flags = (self.data_record_df['Quality Flag 1'] == 0) & (self.data_record_df['Quality Flag 2'] == 0) & (self.data_record_df['Quality Flag 3'] == 0)
            good_flag = check_quality_flags
        
        # Print the fraction of good measurements
        good_ratio = np.round((len(self.data_record_df[good_flag]) / len(self.data_record_df)) * 100, 2)
        print(f"{good_ratio} % good data of {len(self.data_record_df)} spectra, out of {self.metadata.number_of_measurements} measurements")
        
        # Throw away bad data, keep the good, re-assigning and over-writing the existing class attribute
        self.data_record_df = self.data_record_df[good_flag]
        return


    def _delete_intermediate_binary_file(self) -> None:
        os.remove(self.intermediate_file)
        return

    def save_observations(self) -> None:
        """
        Saves the observation data to CSV/HDF5 file and deletes OBR output file.
        """  
        # Create output file name
        outfile = self.intermediate_file.split(".")[0]
        print(f"\nSaving DataFrame to: {outfile}.csv")

        # Save the DataFrame to a file in HDF5 format
        # self.data_record_df.to_hdf(f"{datapath_out}{datafile_out}.h5", key='df', mode='w')
        self.data_record_df.to_csv(f"{outfile}.csv", index=False, mode='w')
        
        # Delete intermediate OBR output file
        self._delete_intermediate_binary_file()
        return
    

    def preprocess_files(self, year: str, month: str, day: str) -> None:
        # Open binary file and extract metadata
        self.open_binary_file()
        
        # Limit observations to specified spatial range
        fields = self.metadata._get_iasi_common_record_fields()
        valid_indices = self.flag_observations_to_keep(fields)

        # Read common IASI record fields and store to pandas DataFrame
        print("\nCommon Record Fields:")
        self.read_record_fields(fields, valid_indices)
        
        if self.data_level == "l1c":
            # Read L1C-specific record fields and add to DataFrame
            print("\nL1C Record Fields:")
            fields = self.metadata._get_iasi_l1c_record_fields()
            self.read_record_fields(fields, valid_indices)
            self.read_spectral_radiance(fields, valid_indices)
            
            # Remove observations (DataFrame rows) based on IASI quality_flags
            self.filter_good_spectra(datetime(int(year), int(month), int(day)))
        
        if self.data_level == "l2":
            # Read L2-specific record fields and add to DataFrame
            print("\nL2 Record Fields:")
            # self.read_record_fields(self.metadata._get_iasi_l2_record_fields())
            
            # # Remove observations (DataFrame rows) based on IASI cloud_phase
            # self.filter_specified_cloud_phase()
        self.close_binary_file()

        # Construct Local Time column
        self.build_local_time()
        # Construct Datetime column and remove individual time elements
        self.build_datetime()
        # Save filtered DataFrame to CSV/HDF5
        self.save_observations()
        
        # Print the DataFrame
        print(self.data_record_df)