import subprocess

from pisco import Configurer

def delete_empty_dirs(path):
    """Deletes empty directories recursively from a given path."""
    command = ["find", path, "-type", "d", "-empty", "-delete"]
    try:
        subprocess.run(command, check=True)
        print("Empty directories deleted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while deleting empty directories: {e}")

def list_non_empty_dirs(path):
    """Lists directories that are not empty, indicating potentially failed jobs."""
    command = ["bash", "-c", f"find {path} -type f | awk -F/ 'OFS=\"/\"{{NF--; print}}' | sort -u"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        non_empty_dirs = result.stdout.strip().split('\n')
        if non_empty_dirs:
            print("List of directories that are not empty:")
            for dir_path in non_empty_dirs:
                print(dir_path)
        else:
            print("No non-empty directories found.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while listing non-empty directories: {e}")

def main():
    """Clean up the pisco run by deleting empty directories and listing non-empty ones."""
    # Instantiate a Configurer class to get data from configuration
    config = Configurer()

    # Path for the find command, constructed from config data
    path = f"{config.datapath}{config.satellite_identifier}"

    # Delete empty directories and list non-empty directories
    delete_empty_dirs(path)
    list_non_empty_dirs(path)

if __name__ == "__main__":
    main()
