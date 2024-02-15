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
    """Prints the last 50 lines of the pisco.log file in the corresponding parent directory."""
    for dir_path in dir_paths:
        # Construct the parent directory path by removing 'l1c' or 'l2'
        parent_dir = os.path.normpath(os.path.join(dir_path, os.pardir, os.pardir))
        log_path = os.path.join(config.datapath, config.satellite_identifier, parent_dir, "pisco.log")
        if os.path.exists(log_path):
            print(f"\nLast 50 lines of the log file in {parent_dir}:")
            try:
                subprocess.run(["tail", "-n", "50", log_path], check=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while reading the log file in {parent_dir}: {e}")
        else:
            print(f"No log file found in {parent_dir}")

def main():
    """Clean up the pisco run by deleting empty directories and listing non-empty ones."""
    config = Configurer()

    temp_paths = [
        os.path.join(config.datapath, config.satellite_identifier, "l1c"),
        os.path.join(config.datapath, config.satellite_identifier, "l2")
    ]

    delete_empty_dirs(temp_paths)
    non_empty_dirs = list_non_empty_dirs(temp_paths)

    print_log_tail(non_empty_dirs, config)

if __name__ == "__main__":
    main()
