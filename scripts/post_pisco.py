import os
import subprocess

from pisco import Configurer

def list_non_empty_dirs(paths):
    """
    Lists directories that are not empty for the given paths.
    
    This function is used to identify directories that potentially contain
    data from failed runs, based on the presence of files.

    Parameters:
    - paths (list): A list of directory paths to search for non-empty directories.

    Returns:
    - list: A list of paths to non-empty directories found within the specified paths.
    """
    non_empty_dirs = []
    for path in paths:
        command = ["bash", "-c", f"find {path} -type f | awk -F/ '{{OFS=\"/\"}}{{NF--; print}}' | sort -u"]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            dirs = result.stdout.strip().split('\n')
            if dirs:  # Only extend if dirs is not empty to avoid adding empty strings
                non_empty_dirs.extend(dirs)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while listing non-empty directories in {path}: {e}")
    return non_empty_dirs

def print_log_tail_interactively(dir_paths, config):
    """
    Interactively prints the last 50 lines of the 'pisco.log' file located in the directories.

    Parameters:
    - dir_paths (list): A list of directory paths where 'pisco.log' files will be searched for.
    - config (Configurer): An instance of the Configurer class with configuration settings.
    """
    print("Do you want to print the tails of the log files? [y/N]: ", end="")
    response = input().strip().lower()
    if response != 'y':
        return

    for dir_path in dir_paths:
        log_path = os.path.join(dir_path, "pisco.log")
        if os.path.exists(log_path):
            print(f"\nInspecting log file: {log_path}")
            try:
                output = subprocess.check_output(["tail", "-n", "10", log_path], text=True)
                print(output)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while reading the log file: {e}")
            print("Press enter to continue to the next log file, or type 'exit' to stop: ", end="")
            if input().strip().lower() == 'exit':
                break
        else:
            print(f"No log file found in: {dir_path}")

def main():
    """
    Main function to list non-empty directories and allow interactive inspection of 'pisco.log' files.

    This function facilitates the identification and examination of potentially failed runs
    by inspecting the corresponding log files.
    """
    config = Configurer()

    # Paths to search for non-empty directories
    paths = [
        os.path.join(config.datapath, config.satellite_identifier, "l1c"),
        os.path.join(config.datapath, config.satellite_identifier, "l2")
    ]

    non_empty_dirs = list_non_empty_dirs(paths)
    print_log_tail_interactively(non_empty_dirs, config)

if __name__ == "__main__":
    main()
