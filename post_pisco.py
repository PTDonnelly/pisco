import subprocess
from pisco import Extractor

def main():
    """Clean up the pisco run:
    
    1. Delete empty directories
    """
    # Location of jsonc configuration file
    path_to_config_file = "inputs/config.jsonc"
    
    # Instantiate an Extractor class to get data from raw binary files
    ex = Extractor(path_to_config_file)

    # Path for the find command
    path = ex.config.datapath

    # Build command to recursively delete empty directories created from the Processing module
    command = ["find", path, "-type", "d", "-empty", "-delete"]

    # Execute on command line
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
