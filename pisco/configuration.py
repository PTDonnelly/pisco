from typing import List
import commentjson
import os

class Configurer:
    def __init__(self, path_to_config_file: str):
        self.data_level: str = ""
        
        # Initialise the Config class with your JSON configuration file
        with open(path_to_config_file, 'r') as file:
            # Access the parameters directly as attributes of the class. 
            self.__dict__ = commentjson.load(file)
            
        # Perform any necessary post-processing before executing
        self.latitude_range, self.longitude_range = tuple(self.latitude_range), tuple(self.longitude_range)
        self.channels: List[int] = None
        self.datapath_out = f"{self.datapath_out}{self.satellite_identifier}/"
        os.makedirs(self.datapath_out, exist_ok=True)
    
    @staticmethod
    def set_channels(mode):
        # Set the list of IASI spectral channel indices
        if mode == "all":
            # Extract all 8461 IASI L1C spectral channels
            return [(i + 1) for i in range(8461)]
        elif mode == "range":
            # Specify a subset of channels
            start_channel = 844
            end_channel = 2220
            return  [i for i in range(start_channel, end_channel+1)]
        elif mode == "flag":
            # Select single channel for fast processing
            return [1]
        else:
            raise ValueError('mode must be "all", "range", or "flag" for L1C reduction')