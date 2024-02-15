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
    """Lists non-empty directories for given paths and returns them."""
    non_empty_dirs = []
    for path in paths:
        command = ["bash", "-c", f"find {path} -type f | awk -F/ '{{OFS=\"/\"}}{{NF--; print}}' | sort -u"]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            dirs = result.stdout.strip().split('\n')
            non_empty_dirs.extend(dirs)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while listing non-empty directories in {path}: {e}")
    return non_empty_dirs

def print_log_tail(dir_paths, config):
    """Prints the last 50 lines of the pisco.log file located in the parent directory of the non-empty l1c or l2 directories."""
    for dir_path in dir_paths:
        # Remove 'l1c' or 'l2' from the dir_path and append 'pisco.log'
        base_dir = dir_path.replace("/l1c", "").replace("/l2", "")
        log_path = os.path.join(base_dir, "pisco.log")

        if os.path.exists(log_path):
            print(f"\nLast 50 lines of the log file in {base_dir}:")
            try:
                subprocess.run(["tail", "-n", "50", log_path], check=True, text=True, capture_output=True)
                output = subprocess.check_output(["tail", "-n", "50", log_path], text=True)
                print(output)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while reading the log file in {base_dir}: {e}")
        else:
            print(f"No log file found in {base_dir}")

def main():
    """Main function to orchestrate the directory clean-up and log file tail printing."""
    config = Configurer()

    # Define paths to l1c and l2 directories
    paths = [
        os.path.join(config.datapath, config.satellite_identifier, "l1c"),
        os.path.join(config.datapath, config.satellite_identifier, "l2")
    ]

    # Obtain a list of non-empty directories within the l1c and l2 paths
    non_empty_dirs = list_non_empty_dirs(paths)

    # Print the last 50 lines of the pisco.log file for each non-empty directory
    print_log_tail(non_empty_dirs, config)

if __name__ == "__main__":
    main()
