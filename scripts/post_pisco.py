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

def print_log_tail_interactively(dir_paths, config):
    """Interactively prints the last 50 lines of the pisco.log file located in the parent directory of the non-empty l1c or l2 directories."""
    print("Do you want to print the tails of the log files? [y/n]: ", end="")
    response = input().strip().lower()
    if response != 'y':
        return

    for dir_path in dir_paths:
        base_dir = dir_path.replace("/l1c", "").replace("/l2", "")
        log_path = os.path.join(base_dir, "pisco.log")

        if os.path.exists(log_path):
            print(f"\nFull path of the log file: {log_path}")
            print(f"Last 50 lines of the log file in {base_dir}:")
            try:
                output = subprocess.check_output(["tail", "-n", "50", log_path], text=True)
                print(output)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while reading the log file in {base_dir}: {e}")
            print("Press enter to continue to the next logfile, or type 'exit' to stop: ", end="")
            if input().strip().lower() == 'exit':
                break
        else:
            print(f"No log file found in {base_dir}")

def main():
    """Main function to orchestrate the directory clean-up and interactive log file tail printing."""
    config = Configurer()

    paths = [
        os.path.join(config.datapath, config.satellite_identifier, "l1c"),
        os.path.join(config.datapath, config.satellite_identifier, "l2")
    ]

    delete_empty_dirs(paths)
    non_empty_dirs = list_non_empty_dirs(paths)

    print_log_tail_interactively(non_empty_dirs, config)

if __name__ == "__main__":
    main()
