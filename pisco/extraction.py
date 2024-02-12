import logging
import os
import subprocess
from typing import List, Tuple

from .configuration import Configurer

# Obtain a logger for this module
logger = logging.getLogger(__name__)

class Extractor:
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
        self.intermediate_file: str = None
        self.intermediate_file_check: bool = None

        
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
            return os.path.join("bdd",self.config.satellite_identifier, self.data_level, "iasi")
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


    def create_intermediate_filepath(self) -> str:
        """
        Creates the directory to save the output files.
        
        Returns:
            intermediate_file (str): the full path to the intermediate file produced by IASI extraction script.
        """
        if self.data_level == 'l1c':
            self.datafile_out = f"extracted_spectra.txt"
        elif self.data_level == 'l2':
            self.datafile_out = f"cloud_products.txt"
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.datapath_out, exist_ok=True)
        return os.path.join(self.datapath_out, self.datafile_out)


    def _build_parameters(self) -> str:
        """
        Builds the parameter string for the IASI data extraction command.
        
        Returns:
            str: Parameters for the command.
        """
        # Define the parameters for the command
        if (self.data_level == 'l1c'):
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
        elif (self.data_level == 'l2'):
            list_of_parameters = [
                f"-d2 {self.datapath_in}", # l2 data directory
                f"-fd {self.year}-{self.month}-{self.day} -ld {self.year}-{self.month}-{self.day}", # first and last day
                f"-mila {self.config.latitude_range[0]} ", # min_latitude
                f"-mala {self.config.latitude_range[1]} ", # max_latitude
                f"-milo {self.config.longitude_range[0]} ", # min_longitude
                f"-malo {self.config.longitude_range[1]} ", # max_longitude
                f"-t2 {self.config.products}", # l2 products
                f"-of txt"  # output file format
            ]
        # Join the parameters into a single string and return
        return ' '.join(list_of_parameters)


    def _get_command(self) -> str:
        """
        Builds the command to extract IASI data based on the data level.

        Raises:
            ValueError: If the data level is neither 'l1c' nor 'l2'.

        Returns:
            str: Command to run for data extraction.
        """
        if (self.data_level == 'l1c') or (self.data_level == 'l2'):
            # Define the path to the run executable
            full_runpath = os.path.join(self.runpath, "bin", "obr_v4")
            # Get the command parameters
            parameters = self._build_parameters()
            # Return the complete command
            return f"{full_runpath} {parameters} -out {self.datapath_out}{self.datafile_out}"
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")


    def run_command(self) -> object:
        """
        Executes and monitors the command to extract IASI data.
        """
        # Build the command string to execute the binary script
        command = self._get_command()

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
                    logging.info(current_output_line.strip())
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
        
        # Create a CompletedProcess object that contains the result of execution
        return subprocess.CompletedProcess(args=command, returncode=return_code, stdout='\n'.join(command_output), stderr=None)

    @staticmethod
    def _get_l2_products_for_file_check(products):
        return products.split(',')


    def check_extracted_files(self, result: object) -> bool:
        products = self._get_l2_products_for_file_check(self.config.products)

        # If binary script runs but detects no data, report back, delete the empty intermediate file, and return False
        if ("No L1C data files found" in result.stdout) or any(f"0 {product} data selected out of 0" in result.stdout for product in products):
            logging.info(result.stdout)
            os.remove(self.intermediate_file)
            return False
        else:
            return True


    def extract_files(self) -> Tuple[bool, str]:
        """
        Preprocesses the IASI data.
        """
        # Create the output directory and point to intermediate file (L1C: OBR, L2: BUFR)
        self.intermediate_file = self.create_intermediate_filepath()
        print(self.intermediate_file)
        exit()
        # Run the command to extract the data
        result = self.run_command()
        # Check if files are produced. If not, skip processing.
        self.intermediate_file_check = self.check_extracted_files(result)