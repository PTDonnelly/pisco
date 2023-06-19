import os
import subprocess
from typing import Optional, Tuple

from .configurer import Configurer

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
        if self.data_level == 'l1c':
            # Format the input path string and return it
            return f"/bdd/metopc/{self.data_level}/iasi/"
        elif self.data_level == 'l2':
            # Format the input path string with an additional 'clp/' at the end and return it
            return f"/bdd/metopc/{self.data_level}/iasi/{self.year}/{self.month}/{self.day}/clp/"
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


    def check_preprocessed_files(self, result: object) -> bool:
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
        list_of_parameters = [
            f"-fd {self.year}-{self.month}-{self.day} -ld {self.year}-{self.month}-{self.day}",  # first and last day
            f"-c {self.config.channels[0]}-{self.config.channels[-1]}",  # spectral channels
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
        if self.data_level == 'l1c':
            # Define the path to the run executable
            runpath = f"./bin/obr_v4"
            # Get the command parameters
            parameters = self._build_parameters()
            # Return the complete command
            return f"{runpath} -d {self.datapath_in} {parameters} -out {self.datapath_out}{self.datafile_out}"
        elif self.data_level == 'l2':
            # Define the path to the run executable
            runpath = "./bin/BUFR_iasi_clp_reader_from20190514"
            # Return the complete command
            return f"{runpath} {self.datapath_in}{self.datafile_in} {self.datapath_out}{self.datafile_out}"
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")

    def run_command(self) -> Optional[bool]:
        """
        Executes and monitors the command to extract IASI data.
        """
        # Build the command string to execute the binary script
        command = self._get_command()
        print(command)
        try:
            # Run the command in a bash shell and capture the output
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # The subprocess module will raise a CalledProcessError if the process returns a non-zero exit status
            # The standard error of the command is available in e.stderr
            raise RuntimeError(f"{str(e)}, stderr: {e.stderr.decode('utf-8')}")
        except Exception as e:
            # Catch any other exceptions
            raise RuntimeError(f"An unexpected error occurred while running the command '{command}': {str(e)}")
        return result


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
            # self.datafile_out = "cloud_products.csv"
            self.datafile_out = self.datafile_in.split(",")[2]
        else:
            # If the data level is not 'l1c' or 'l2', raise an error
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.datapath_out, exist_ok=True)
        return f"{self.datapath_out}{self.datafile_out}"


    def preprocess(self) -> Tuple[bool, str]:
        """
        Preprocesses the IASI data.
        """
        # Create the output directory and point to intermediate file (L1C: OBR, L2: BUFR)
        self.intermediate_file = self.create_intermediate_filepath()
        # Run the command to extract the data
        result = self.run_command()
        # Check if files are produced. If not, skip processing.
        self.intermediate_file_check = self.check_preprocessed_files(result)
                        

    def _get_suffix(self):
        old_suffix=".bin"
        if self.data_level == 'l1c':
            new_suffix=".bin"
        elif self.data_level == 'l2':
            new_suffix=".out"
        else:
            raise ValueError("Invalid data path type. Accepts 'l1c' or 'l2'.")
        return old_suffix, new_suffix

    def rename_files(self):
        old_suffix, new_suffix = self._get_suffix()
        if os.path.isdir(self.datapath_out):
            for filename in os.scandir(self.datapath_out):
                if filename.name.endswith(old_suffix):
                    new_filename = f"{filename.name[:-len(old_suffix)]}{new_suffix}"
                    os.rename(filename.path, os.path.join(self.datapath_out, new_filename))
