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


def print_log_tail(dir_paths, log_filename="run.log"):
    """Prints the last 50 lines of the log file in each given directory."""
    for dir_path in dir_paths:
        log_path = os.path.join(dir_path, log_filename)
        if os.path.exists(log_path):
            print(f"\nLast 50 lines of the log file in {dir_path}:")
            try:
                subprocess.run(["tail", "-n", "50", log_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while reading the log file in {dir_path}: {e}")
        else:
            print(f"No log file found in {dir_path}")


def main():
    """Clean up the pisco run by deleting empty directories and listing non-empty ones."""
    # Instantiate a Configurer class to get data from config.json
    config = Configurer()

    # Paths for the find command, constructed from config data
    paths = [
        os.path.join(config.datapath, config.satellite_identifier, "l1c"),
        os.path.join(config.datapath, config.satellite_identifier, "l2")
        ]

    # Delete empty directories, then list non-empty directories and print the tail of their log files
    delete_empty_dirs(paths)
    non_empty_dirs = list_non_empty_dirs(paths)
    print_log_tail(non_empty_dirs)

if __name__ == "__main__":
    main()
