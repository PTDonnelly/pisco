import os
import subprocess
from typing import Optional, Tuple

from .configuration import Configurer

class Extractor:
    def __init__(self, path_to_config_file: str):
        """
        Initialize the Extractor class with given parameters.

        Args:
           path_to_config_file (str): Location of jsonc configuration file
        """
        # Instantiate the Config class and set_parameters() for analysis
        self.config = Configurer(path_to_config_file)
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
            return f"{self.config.datapath_out}{self.data_level}/{self.year}/{self.month}/{self.day}/"
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
            return f"/bdd/metopc/{self.data_level}/iasi/"
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


    def check_extracted_files(self, result: object) -> bool:
        # If binary script runs but detects no data, report back, delete the empty intermediate file, and return False
        if ("No L1C data files found" in result.stdout) or ("No L2 data files found" in result.stdout):
            print(result.stdout)
            os.remove(self.intermediate_file)
            return False
        else:
            return True


    def _build_parameters(self) -> str:
        """
        Builds the parameter string for the IASI data extraction command.
        
        Returns:
            str: Parameters for the command.
        """
        # Define the parameters for the command
        if (self.data_level == 'l1c'):
            list_of_parameters = [
                f"-d {self.datapath_in}", # l1c data directory
                f"-fd {self.year}-{self.month}-{self.day} -ld {self.year}-{self.month}-{self.day}",  # first and last day
                f"-c {self.config.channels[0]}-{self.config.channels[-1]}",  # spectral channels
                f"-of bin"  # output file format
            ]
        elif (self.data_level == 'l2'):
            list_of_parameters = [
                f"-d2 {self.datapath_in}", # l2 data directory
                f"-fd {self.year}-{self.month}-{self.day} -ld {self.year}-{self.month}-{self.day}", # first and last day
                f"-t2 {self.config.products}", # l2 products
                f"-of bin"  # output file format
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
            runpath = f"./bin/obr_v4"
            # Get the command parameters
            parameters = self._build_parameters()
            # Return the complete command
            return f"{runpath} {parameters} -out {self.datapath_out}{self.datafile_out}"
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")

    def run_command(self) -> Optional[bool]:
        """
        Executes and monitors the command to extract IASI data.
        """
        # Build the command string to execute the binary script
        command = self._get_command()
        print(f"\n{command}")

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
                    print(current_output_line.strip())
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



    def create_intermediate_filepath(self) -> None:
        """
        Creates the directory to save the output files, based on the input file name and time.
        
        Returns:
            intermediate_file (str): the full path to the intermediate file produced by IASI extraction script.
        """
        if self.data_level == 'l1c':
            # Get the output file name from the input file name
            self.datafile_out = "extracted_spectra.bin"
        elif self.data_level == 'l2':
            self.datafile_out = "cloud_products.bin"
            # self.datafile_out = self.datafile_in.split(",")[2]
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.datapath_out, exist_ok=True)
        return f"{self.datapath_out}{self.datafile_out}"


    def extract_files(self) -> Tuple[bool, str]:
        """
        Preprocesses the IASI data.
        """
        # Create the output directory and point to intermediate file (L1C: OBR, L2: BUFR)
        self.intermediate_file = self.create_intermediate_filepath()
        # Run the command to extract the data
        result = self.run_command()
        # Check if files are produced. If not, skip processing.
        self.intermediate_file_check = self.check_extracted_files(result)