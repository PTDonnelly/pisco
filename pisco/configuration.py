from typing import List
import commentjson

class Configurer:
    def __init__(self, path_to_config_file: str):
        self.data_level: str = ""
        
        # Initialise the Config class with your JSON configuration file
        with open(path_to_config_file, 'r') as file:
            # Access the parameters directly as attributes of the class. 
            self.__dict__ = commentjson.load(file)
            
        # Perform any necessary post-processing before executing
        self.latitude_range, self.longitude_range = tuple(self.latitude_range), tuple(self.longitude_range)
        self.channels: List[int] = self.set_channels(self.channel_mode)
    
    @staticmethod
    def set_channels(mode):
        # Set the list of IASI spectral channel indices
        if mode == "all":
            # Defaults to maximum of 8461 channels
            return [(i + 1) for i in range(8461)]
        elif mode == "range":
            n = 1
            return [(i + 1) for i in range(n)]
        else:
            raise ValueError('mode but be "all" or "range" for L1C reduction')