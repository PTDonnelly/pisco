from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import subprocess
from typing import List, Tuple, Optional, Dict

from .configuration import Configurer

# Obtain a logger for this module
logger = logging.getLogger(__name__)

class Extractor:
    """Handles the extraction of IASI data for further analysis.

    This class manages the extraction process of IASI (Infrared Atmospheric Sounding Interferometer) data
    based on the configured parameters. It supports the extraction of Level 1C (L1C) and Level 2 (L2) data,
    organizing file paths, and running the necessary extraction commands.

    Attributes:
        config (Configurer): An instance of the Configurer class to access configuration settings.
        runpath (str): The current working directory where the script is executed.
        channels (List[int]): List of spectral channels to be extracted.
        data_level (str): The level of data to be extracted ('l1c' or 'l2').
        year (str): Year of the data to be extracted.
        month (str): Month of the data to be extracted.
        day (str): Day of the data to be extracted.
        datapath_in (str): Input data path.
        datapath_out (str): Output data path where extracted files will be stored.
        datafile_in (str): Name of the input data file.
        datafile_out (str): Name of the output data file.
        intermediate_file_path (str): Path to the intermediate file produced by the extraction process.
        intermediate_file_check (bool): Flag indicating whether the intermediate file has been successfully produced.

    Methods:
        _get_datapath_out(): Determines the output data path based on data level and date.
        _get_datapath_in(): Determines the input data path based on data level.
        get_datapaths(): Retrieves both input and output data paths.
        build_intermediate_file_path(): Creates the path for the intermediate file and ensures the output directory exists.
        _build_parameters(): Constructs the parameter string for the extraction command.
        _get_command(): Builds the full command to be executed for data extraction.
        run_command(): Executes the extraction command and handles its output.
        check_extracted_files(result): Checks the extraction process's result and verifies the intermediate file's presence.
        extract_files(): Orchestrates the data extraction process, including running the extraction command and checking the output.
    """
    def __init__(self):
        """
        Initialize the Extractor class with given parameters.
        """
        # Instantiate a Configurer and set parameters for analysis
        self.config = Configurer()
        self.runpath: str = os.getcwd()
        self.channels: List[int] = None
        self.data_level: str = None
        self.year: str = None
        self.month: str = None
        self.day: str = None
        self.datapath_in: str = None
        self.datapath_out: str = None
        self.datafile_in: str = None
        self.datafile_out: str = None
        self.intermediate_file_path: str = None
        self.intermediate_file_check: bool = None

    @staticmethod
    def _get_common_fields() -> List[Tuple]:
        # Format of OBR fields (field_name, data_type)
        fields = [
            ('SatelliteIdentifier', 'uint32'),
            ('Tb', 'uint8'),
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
        # Format of OBR fields (field_name, data_type)
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
    def _get_l1c_product_fields(channels: List[int]) -> List[Tuple]:
        # Format of spectral fields (field_name, data_type) where field_name is the channel ID (values are radiance in native IASI units of mW.m-2.st-1.(cm-1)-1)
        fields = [(str(channel_id), 'float32') for channel_id in channels]
        return fields
      
    @staticmethod
    def _get_l2_product_fields() -> List[Tuple]:
        # Format of OBR fields (field_name, data_type)
        fields = [
            ('Datetime', 'float32'),
            ('Latitude', 'float32'),
            ('Longitude', 'float32'),
            ('Cloud Top Pressure 1', 'float32'),
            ('Temperature or Dry Bulb Temperature 1', 'float32'),
            ('Cloud Amount in Segment 1', 'float32'),
            ('Cloud Phase 1', 'uint32'),
            ('Cloud Top Pressure 2', 'float32'),
            ('Temperature or Dry Bulb Temperature 2', 'float32'),
            ('Cloud Amount in Segment 2', 'float32'),
            ('Cloud Phase 2', 'uint32'),
            ('Cloud Top Pressure 3', 'float32'),
            ('Temperature or Dry Bulb Temperature 3', 'float32'),
            ('Cloud Amount in Segment 3', 'float32'),
            ('Cloud Phase 3', 'uint32')
            ]
        return fields
    

    def _get_fields_and_datatypes(self) -> Dict[str, str]:
        # Read and combine byte tables to optimise reading of OBR txtfile
        if self.data_level == 'l1c':
            combined_fields = (
                Extractor._get_common_fields() +
                Extractor._get_l1c_record_fields() +
                Extractor._get_l1c_product_fields(self.channels)
                )
        if self.data_level == 'l2':
            combined_fields = (
                Extractor._get_l2_product_fields()
                )

        # Create dtype dict from combined fields
        return {field[0]: field[1] for field in combined_fields}


    def _get_datapath_out(self) -> str:
        """
        Gets the data path for the output based on the data level, year, month, and day.

        Raises:
            ValueError: If the data level is neither 'l1c' nor 'l2'.
            
        Returns:
            str: Output data path.
        """
        if (self.data_level == 'l1c') or (self.data_level == 'l2'):
            return os.path.join(self.config.datapath, self.config.satellite_identifier, self.data_level, self.year, self.month, self.day)
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")


    def _get_datapath_in(self) -> str:
        """
        Gets the data path for the input based on the data level, year, month, and day.
        
        Raises:
            ValueError: If the data level is neither 'l1c' nor 'l2'.
            
        Returns:
            str: Input data path.
        """
        # Check the data level
        if (self.data_level == 'l1c') or (self.data_level == 'l2'):
            # Format the input path string and return it
            if int(self.year) < 2013:
                return os.path.join("/bdd", "IASI", self.data_level.upper(), {self.year}, {self.month}, {self.day}, self.config.products)
            else:
                return os.path.join("/bdd",self.config.satellite_identifier, self.data_level, "iasi")
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")


    def get_datapaths(self) -> None:
        """
        Retrieves the input and output data paths.
        """
        # Get the input data path
        self.datapath_in = self._get_datapath_in()
        # Get the output data path
        self.datapath_out = self._get_datapath_out()
        return


    def build_intermediate_file_path(self, date_time: Optional[str]=None) -> str:
        """
        Creates the directory to save the output files.
        
        Returns:
            intermediate_file (str): the full path to the intermediate file produced by IASI extraction script.
        """
        if self.data_level == 'l1c':
            self.datafile_out = f"extracted_spectra.txt"
        elif self.data_level == 'l2':
            self.datafile_out = f"{date_time}_cloud_products.txt" if date_time else "cloud_products.txt"
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.datapath_out, exist_ok=True)
        return os.path.join(self.datapath_out, self.datafile_out)


    def get_raw_l2_product_files(self) -> List[Path]:
        # Define the full path to the location of the binary L2 files
        l2_product_directory = Path(self.datapath_in) / self.year / self.month / self.day / self.config.products

        # Scan for all files in the directory
        l2_product_files = list(l2_product_directory.glob('*.bin'))
        return l2_product_files
    

    def get_reduced_l2_product_files(self) -> List[Path]:
        # Define the full path to the location of the text files extracted from the binary L2 files
        l2_product_directory = Path(self.datapath_out)

        # Scan for all files in the directory
        l2_product_files = list(l2_product_directory.glob('*.txt'))
        return l2_product_files

    
    def _build_parameters(self) -> str:
        """
        Builds the parameter string for the IASI data extraction command.
        
        Returns:
            str: Parameters for the command.
        """
        # Define the parameters for the command
        if self.data_level == 'l1c':
            # Set range of spectral channels to use (pass custom spectral range as arguments, defaults to channels 220-2220 in the main absorption band of water ice)
            self.channels = self.config.set_channels(self.config.channels_mode)
            # Create a string of all channel IDs separated by commas
            channel_map = ",".join(map(str, self.channels))
            
            list_of_parameters = [
                f"-d {self.datapath_in}", # l1c data directory
                f"-fd {self.year}-{self.month}-{self.day} -ld {self.year}-{self.month}-{self.day}",  # first and last day
                f"-mila {self.config.latitude_range[0]} ", # min_latitude
                f"-mala {self.config.latitude_range[1]} ", # max_latitude
                f"-milo {self.config.longitude_range[0]} ", # min_longitude
                f"-malo {self.config.longitude_range[1]} ", # max_longitude
                f"-c {channel_map}",  # spectral channels
                f"-qlt {self.config.quality_flags}",
                f"-of txt"  # output file format
            ]

        # Join the parameters into a single string and return
        return ' '.join(list_of_parameters)


    def _get_version_from_file_path(self, satellite: str, date_time: datetime) -> int:
        # Convert the datetime string to a datetime object
        measurement_date_time = datetime.strptime(date_time, "%Y%m%d%H%M%S")

        # Define the cutoff datetimes and clp reader version for MetOp satellites A and B (C is most recent and uses version 6 by default)
        cutoffs = {
            'a': [
                ("20140930072357", 1),
                ("20150702071153", 2),
                ("20150924084159", 3),
                ("20170620083857", 4),
                ("20190514071758", 5),
            ],
            'b': [
                ("20140930081455", 1),
                ("20150702140854", 2),
                ("20150924095658", 3),
                ("20170620093255", 4),
                ("20190514080259", 5),
            ]
        }
        
        # Ensure satellite letter is lowercase for matching keys in the dictionary
        satellite = satellite.lower()
        if satellite not in cutoffs:
            raise ValueError("Satellite must be 'a', 'b', or 'c'.")

        # Iterate through the cutoffs to find the correct version
        for cutoff_str, version in cutoffs[satellite]:
            cutoff_datetime = datetime.strptime(cutoff_str, "%Y%m%d%H%M%S")
            if measurement_date_time < cutoff_datetime:
                return version
            
        # If none of the conditions were met, it means the date is after the last cutoff
        return 6


    def _get_clp_version(self, file: Path) -> int:
        """Extract the satellite identifier, date, and time from the file name"""
        if int(self.year) < 2013:
            pass
        else:
            pattern = re.compile(r'METOP([ABC])\+IASI_C_EUMP_(\d{14})')
            match = pattern.search(file.name)
            if not match:
                raise ValueError("File name format does not match expected pattern.")

        satellite, date_time = match.groups()

        # Use the extracted information to determine the version
        return self._get_version_from_file_path(satellite, date_time), date_time


    def get_command(self) -> str:
        """
        Builds the command to extract IASI data based on the data level.

        Raises:
            ValueError: If the data level is neither 'l1c' nor 'l2'.

        Returns:
            str: Command to run for data extraction.
        """
        if self.data_level == 'l1c':
            # Define the path to the run executable
            executable_runpath = os.path.join(self.runpath, "bin", "obr_v4")
            # Get the command parameters
            parameters = self._build_parameters()
            # Create the output directory and point to intermediate file
            self.intermediate_file_path = self.build_intermediate_file_path()
            # Return the complete command
            return f"{executable_runpath} {parameters} -out {self.intermediate_file_path}"
        
        elif self.data_level == 'l2':
            # Get version of IASI L2 CLP reader based on the date and time of observation
            version, date_time = self._get_clp_version(self.datafile_in)
            # Create the output directory and point to intermediate file
            self.intermediate_file_path = self.build_intermediate_file_path(date_time)
            # Define the path to the run executable
            executable_runpath = os.path.join(self.runpath, "bin",  "clpall_ascii")
            # Return the complete command
            return f"{executable_runpath} {self.datafile_in} {self.intermediate_file_path} v{version}"
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")


    def run_command(self, command: str) -> object:
        """
        Executes and monitors the command to extract IASI data.
        """
        try:
            # Initiate the subprocess with Popen.
            # shell=True specifies that the command will be run through the shell.
            # stdout=subprocess.PIPE and stderr=subprocess.PIPE allow us to capture the output.
            # text=True means the output will be in string format (if not set, the output would be in bytes).
            subprocess_instance = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Initialize an empty list to store the output lines
            command_output = []

            # Loop until the subprocess finishes.
            # We're using subprocess_instance.poll() to check if the subprocess has finished (it returns None while the subprocess is still running).
            while True:
                # Read one line of output. If there's no output, readline() returns an empty string.
                current_output_line = subprocess_instance.stdout.readline()
                if subprocess_instance.poll() is not None:
                    break
                if current_output_line:
                    # Print the line and also save it to command_output list
                    logger.info(current_output_line.strip())
                    command_output.append(current_output_line.strip())

            # At this point, the subprocess has finished. Check its return code.
            return_code = subprocess_instance.poll()

            # If the return code is not 0, it usually indicates an error.
            if return_code != 0:
                # If there's an error, read the error message and raise an exception.
                error_message = subprocess_instance.stderr.read()
                raise RuntimeError(f"Command '{command}' returned non-zero exit status {return_code}, stderr: {error_message}")
        except Exception as unexpected_error:
            # Catch any other exceptions that weren't handled above.
            raise RuntimeError(f"An unexpected error occurred while running the command '{command}': {str(unexpected_error)}")
        return


    def _delete_intermediate_file(self, filepath) -> None:
        """Deletes the specified intermediate file."""
        try:
            os.remove(filepath)
        except OSError as e:
            logger.error(f"Error deleting file: {e}")


    def build_converters(self) -> dict:        
        # Create dtype dict from combined fields
        dtype_dict = self._get_fields_and_datatypes()
        
        # Create converters based on dtype_dict
        converters = {}
        for column, dtype in dtype_dict.items():
            if dtype == float:
                # For float columns, replace ' -nan' with np.nan and try converting to float
                converters[column] = lambda x: np.nan if x.strip() in ['-nan', ''] else float(x)
            elif dtype == int:
                # For int columns, you might want to handle NaN differently since int types don't support np.nan
                converters[column] = lambda x: None if x.strip() in ['-nan', ''] else int(x)
            else:
                # For other types, just strip whitespace (customize as needed)
                converters[column] = lambda x: x.strip()
        return converters
    

    def apply_converters_to_df(self, converters, df):
        # Apply each converter function to its respective column in the DataFrame
        for column, func in converters.items():
            # Ensure column refers to existing DataFrame column; adjust if using numerical indices or actual names
            if column in df.columns:
                df[column] = df[column].apply(func)

        return df

    def process_file(self, file, converters):
        # Read each intermediate text file into a DataFrame
        df = pd.read_csv(file, header=None, names=['Data'])

        # Split single-column string into separate columns of strings
        df_expanded = df['Data'].str.split(expand=True)
        df_expanded.columns = converters.keys()

        # Set data types of columns using converter functions
        df_expanded = self.apply_converters_to_df(converters, df_expanded)
        return df_expanded

    def combine_files(self):
        logger.info(f"Combining L2 cloud products")

        # Build data type converter functions to account for NaNs
        converters = self.build_converters()

        # Get paths of individual files as Path() objects
        files = self.get_reduced_l2_product_files()

        # Use ProcessPoolExecutor to parallelize file processing
        df_list = []
        with ProcessPoolExecutor() as executor:
            # Submit all file processing tasks and execute them in parallel
            future_to_file = {executor.submit(self.process_file, file, converters): file for file in files}
            
            for future in as_completed(future_to_file):
                df_expanded = future.result()
                # Append DataFrame to list and delete text file
                df_list.append(df_expanded)
                self._delete_intermediate_file(file)


        # # Initialize an empty list to store DataFrames
        # df_list = []
        # for file in files:
            
        #     # Read each intermediate text file into a DataFrame
        #     df = pd.read_csv(file, header=None, names=['Data'])

        #     # Split single-column string into separate columns of strings
        #     df_expanded = df['Data'].str.split(expand=True)
        #     df_expanded.columns = converters.keys()

        #     # Set data types of columns using converter functions
        #     df_expanded = self.apply_converters_to_df(converters, df_expanded)

        #     # Append DataFrame to list and delete text file
        #     df_list.append(df_expanded)
        #     self._delete_intermediate_file(file)
        
        # Concatenate all DataFrames along the rows (axis=0)
        combined_df = pd.concat(df_list, axis=0)
        # Sort the DataFrame based on the "Datetime" column
        combined_df.sort_values(by="Datetime", inplace=True)
        # Reset index if you want a clean, continuous index
        combined_df.reset_index(drop=True, inplace=True)

        # Write the combined DataFrame to a new CSV file
        logger.info(f"Saving daily combined L2 cloud products: {self.intermediate_file_path}")
        self.intermediate_file_path = self.build_intermediate_file_path()
        combined_df.to_csv(self.intermediate_file_path, sep='\t', index=False)
        return


    def check_extracted_file(self):
        # Create a Path object
        file_path = Path(self.intermediate_file_path)
        
        # Check if the file exists and is non-empty
        if file_path.exists() and file_path.stat().st_size > 0:
            logger.error("Intermediate file exists and is non-empty.")
            return True
        else:
            logger.error("Intermediate file does not exist or is empty.")
            return False


    def extract_files(self) -> Tuple[bool, str]:
        """
        Preprocesses the IASI data.
        """
        # Build the command string to execute the binary script
        command = self.get_command()
        
        # Run the command to extract the data
        self.run_command(command)
        
        return