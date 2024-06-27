import commentjson
import os

class Configurer:
    """Configures and initializes parameters for data processing from a JSON configuration file.

    This class reads a configuration file and initializes various processing parameters.
    It also supports setting the range of spectral channels based on the specified mode.

    Attributes:
        path_to_config_file (str): Path to the JSON configuration file.
        latitude_range (Tuple[float, float]): Tuple representing the minimum and maximum latitude range for data extraction.
        longitude_range (Tuple[float, float]): Tuple representing the minimum and maximum longitude range for data extraction.
        satellite_identifier (str): Identifier for the satellite to process data from.
        datapath (str): Base path where data files are stored or will be stored.
        Other attributes as defined in the configuration file.

    Methods:
        set_channels(mode, start_channel=220, end_channel=2220): Determines the list of spectral channels to use based on the specified mode. Modes include 'all', 'all_reduced', 'range', and 'flag'.

    Raises:
        ValueError: If the mode is 'range' and either start_channel or end_channel is not specified, or if an invalid mode is provided.
    """
    def __init__(self, path_to_config_file: str="inputs/config.jsonc"):
        self.path_to_config_file = path_to_config_file

        # Initialise the Config class with your JSON configuration file
        with open(self.path_to_config_file, 'r') as file:
            # Access the parameters directly as attributes of the class. 
            self.__dict__ = commentjson.load(file)
            
        # Perform any necessary post-processing before executing
        self.latitude_range = tuple(self.latitude_range)
        self.longitude_range = tuple(self.longitude_range)
        os.makedirs(os.path.join(self.datapath, self.satellite_identifier), exist_ok=True)
        
        return
    
    @staticmethod
    def set_channels(mode, start_channel=220, end_channel=2220):
        # Set the list of IASI spectral channel indices
        if mode == "all":
            # Directly return the list comprehension for all 8461 channels
            return list(range(1, 8462))
        elif mode == "all_reduced":
            # Use list slicing to get every second channel starting from the first
            return list(range(1, 8462, 2))
        elif mode == "range":
            if start_channel is None or end_channel is None:
                raise ValueError("Start and end channels must be specified for 'range' mode")
            return list(range(start_channel, end_channel + 1))  # Include end_channel in the range
        elif mode == "flag":
            # Select single channel for fast processing
            return [1]
        else:
            raise ValueError('mode must be "all", "all_reduced", "range", or "flag" for L1C reduction')