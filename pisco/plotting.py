import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

class Plotter:
    """
    Class to contain useful plotting functions for the IASI dataset
    """
    def __init__(self, datapath: str):
        """
        Initializes the Plotter class with a given data path.

        Args:
            datapath (str): The path to the data directory.
        """
        self.datapath = datapath
        self.files_by_date: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    
    def organize_files_by_date(self) -> None:
        """
        Organizes .csv files in the data directory by date.

        The date is inferred from the directory structure: year/month/day.
        The result is stored in self.files_by_date, which is a dictionary
        mapping from (year, month, day) tuples to lists of file paths.

        This creates a dictionary with keys as dates (year, month, day) and values as lists of files.
        """
        for root, dirs, files in os.walk(self.datapath):
            for file in files:
                if ".csv" in file:
                    # Split the root directory path and get year, month and day
                    dir_structure = os.path.normpath(root).split(os.sep)
                    year, month, day = dir_structure[-3], dir_structure[-2], dir_structure[-1]

                    # Append the file path to the corresponding date
                    self.files_by_date[(year, month, day)].append(os.path.join(root, file))

    
    def select_files(self, target_year: str, target_month: str, target_days: List[str], target_file_part: Optional[str] = None) -> List[str]:
        """
        Selects files from the dictionary created by organize_files_by_date method
        based on a target year, month, days and file name part.

        Args:
            target_year (str): The target year as a string.
            target_month (str): The target month as a string.
            target_days (List[str]): The target days as a list of strings.
            target_file_part (Optional[str]): The target part of the file name to select (defaults to None, the file containing all measurements)

        Returns:
            List[str]: List of the file paths that matched the conditions.
        """
        selected_files = []

        # Iterate through dictionary keys
        for (year, month, day), files in self.files_by_date.items():
            # Check if the year, month and day match your conditions
            if year == target_year and month == target_month and day in target_days:
                # Iterate through the files for this date
                for file in files:
                    # Check if the file name contains the target part
                    if target_file_part == None:
                        # Select file containing all measurements
                        selected_files.append(file)
                    elif target_file_part in file:
                        # Select file containing specified measurements
                        selected_files.append(file)

        return selected_files