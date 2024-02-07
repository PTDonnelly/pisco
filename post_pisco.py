import subprocess
from pisco import Configurer

def main():
    """Clean up the pisco run:
    
    1. Delete empty directories
    2. Print the list of directories that are not empty
    """
    # Location of jsonc configuration file
    path_to_config_file = "inputs/config.jsonc"
    
    # Instantiate an Extractor class to get data from raw binary files
    config = Configurer(path_to_config_file)

    # Path for the find command
    path = f"{config.datapath}{config.satellite_identifier}"

    # Build command to recursively delete empty directories created from the Processing module
    delete_empty_dirs_command = ["find", path, "-type", "d", "-empty", "-delete"]

    # Execute command to delete empty directories
    try:
        subprocess.run(delete_empty_dirs_command, check=True)
        print("Empty directories deleted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while deleting empty directories: {e}")

    # Build command to list directories that are not empty
    list_non_empty_dirs_command = ["find", path, "-type", "d", "!", "-empty"]

    # Execute command to list non-empty directories
    try:
        result = subprocess.run(list_non_empty_dirs_command, check=True, capture_output=True, text=True)
        non_empty_dirs = result.stdout.strip().split('\n')
        if non_empty_dirs:
            print("List of directories that are not empty:")
            for dir_path in non_empty_dirs:
                print(dir_path)
        else:
            print("No non-empty directories found.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while listing non-empty directories: {e}")

if __name__ == "__main__":
    main()
