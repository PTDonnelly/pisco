import os
import subprocess

from pisco import Configurer

def delete_empty_dirs(paths):
    """Deletes empty directories recursively from given paths."""
    for path in paths:
        command = ["find", path, "-type", "d", "-empty", "-delete"]
        try:
            subprocess.run(command, check=True)
            print(f"Empty directories deleted successfully in {path}.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while deleting empty directories in {path}: {e}")

def list_non_empty_dirs(paths):
    """Lists non-empty directories for given paths."""
    for path in paths:
        command = ["bash", "-c", f"find {path} -type f | awk -F/ '{{OFS=\"/\"}}{{NF--; print}}' | sort -u"]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            non_empty_dirs = result.stdout.strip().split('\n')
            if non_empty_dirs:
                print(f"List of directories that are not empty in {path}:")
                for dir_path in non_empty_dirs:
                    print(dir_path)
            else:
                print(f"No non-empty directories found in {path}.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while listing non-empty directories in {path}: {e}")

def main():
    """Clean up the pisco run by deleting empty directories and listing non-empty ones."""
    # Instantiate a Configurer class to get data from config.json
    config = Configurer()

    # Paths for the find command, constructed from config data
    paths = [os.path.join(config.datapath, config.satellite_identifier, "l1c"), os.path.join(config.datapath, config.satellite_identifier, "l2")]

    # Delete empty directories and list non-empty directories
    delete_empty_dirs(paths)
    list_non_empty_dirs(paths)

if __name__ == "__main__":
    main()
