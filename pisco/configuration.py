import commentjson
import os

class Configurer:
    def __init__(self, path_to_config_file: str):
        self.data_level: str = ""
        self.datapath_out: str = None

        # Initialise the Config class with your JSON configuration file
        with open(path_to_config_file, 'r') as file:
            # Access the parameters directly as attributes of the class. 
            self.__dict__ = commentjson.load(file)
            
        # Perform any necessary post-processing before executing
        self.latitude_range, self.longitude_range = tuple(self.latitude_range), tuple(self.longitude_range)
        self.datapath_out = f"{self.datapath}{self.satellite_identifier}/"
        os.makedirs(self.datapath_out, exist_ok=True)
    
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