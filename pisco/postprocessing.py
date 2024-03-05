import datetime
import logging
import numpy as np
import os
import re
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Union

from pisco import Configurer, Processor

# Obtain a logger for this module
logger = logging.getLogger(__name__)

class Postprocessor:
    """Processes and analyzes atmospheric data post-extraction.

    This class is designed to further process and analyze extracted atmospheric data.
    It is modular and designed to be changed. It currently supports data organization, selection based on dates,
    and calculation of target variables (such as OLR and cloud phase fractions).

    Attributes:
        filepath (str): Path to the file containing the extracted data.
        cloud_phase_names (Dict[int, str]): Mapping of cloud phase integer codes to descriptive names.
        df (pd.DataFrame or None): DataFrame containing the processed data.
        df_binned (pd.DataFrame or None): DataFrame containing the processed data, downsampled and averaged onto coarse spatial grid.

    Methods:
        get_target_time_range(config): Determines the target time range for analysis based on configuration.
        organise_files_by_date(datapath): Organizes .pkl.gz files in the data directory by date.
        _format_target_date_range(target_range): Formats elements of the target date range as strings.
        select_files(target_range, files_by_date): Selects files based on a target date range from the organized files.
        extract_date_from_filepath(filepath): Extracts the date from a file path using a regular expression.
        _get_dataframe(filepath): Retrieves a DataFrame from a given filepath.
        prepare_dataframe(): Prepares the DataFrame for analysis by filtering and checking its contents.
        set_as_invalid(): Returns a structure indicating invalid data.
        get_outgoing_longwave_radiation(): Calculates OLR values for all cloud phase conditions.
        get_phase_fraction(): Calculates the fraction of measurements for each cloud phase.
        process_target_variables(target_variables, data_dict): Processes each target variable and appends the results to a data dictionary.
        save_results(data_dict, dates, datapath): Saves the processed results into CSV files.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.cloud_phase_names = Postprocessor.get_cloud_phase_dict()
        self.df: pd.DataFrame = None
        self.df_binned: pd.DataFrame = None
    
    @staticmethod
    def get_cloud_phase_dict():
        return {-1: "Clear", 1: "Water", 2: "Ice", 3: "Mixed"}
    
    @staticmethod
    def get_target_time_range(config: Configurer):
        day_list = config.day_list if not config.day_list == "all" else range(1, 32)
        return (config.year_list, config.month_list, day_list)
    
    @staticmethod
    def organise_files_by_date(datapath) -> Dict[Tuple[str, str, str], List[str]]:
        """
        Organises .pkl.gz files in the data directory by date.

        The date is inferred from the directory structure: year/month/day.
        The result is stored in self.files_by_date, which is a dictionary
        mapping from (year, month, day) tuples to lists of file paths.

        This creates a dictionary with keys as tuples of dates (year, month, day) and values as lists of strings of files.
        """
        
        files_by_date = defaultdict(list)  # Initializes an empty list for new keys automatically
        for root, dirs, files in os.walk(datapath):
            for file in files:
                if file.endswith(".pkl.gz"):
                    # Split the root directory path and get year, month and day
                    dir_structure = os.path.normpath(root).split(os.sep)
                    year, month, day = dir_structure[-3], dir_structure[-2], dir_structure[-1]

                    # Append the file path to the corresponding date
                    files_by_date[(year, month, day)].append(os.path.join(root, file))

        return files_by_date  # Convert defaultdict back to dict if necessary

    @staticmethod
    def _format_target_date_range(target_range: Tuple[List]) -> Tuple[List]:
        # Format each element as a string to make YYYY-mm-dd
        formatted_years = [f"{year}" for year in target_range[0]]  # Year formatting remains the same
        formatted_months = [f"{month:02d}" for month in target_range[1]]
        formatted_days = [f"{day:02d}" for day in target_range[2]]
        return (formatted_years, formatted_months, formatted_days)
    

    def select_files(target_range: Tuple[List], files_by_date: Dict[Tuple[str, str, str], List[str]]) -> List[str]:
        """
        Selects files from the dictionary created by organise_files_by_date method
        based on a target year, month, days and file name part.

        Args:
            target_range Tuple[List]: The target date range
            target_months (str): The target month as a string.

        Returns:
            List[str]: List of the file paths that matched the conditions.
        """
        # Format the date range as YYYY-mm-dd
        years, months, days = Postprocessor._format_target_date_range(target_range)

        # Create empty list to store filepaths
        selected_files = []

        # Iterate through dictionary keys
        for (year, month, day), files in files_by_date.items():
            # Check if the year, month and day match your conditions
            if (year in years) and (month in months) and (day in days):
                # Iterate through the files for this date
                for file in files:
                    # Select file containing all measurements
                    selected_files.append(file)
        return sorted(selected_files)
    
    @staticmethod
    def extract_date_from_filepath(filepath: str) -> object:
        """
        Extracts the date from a file path using a regular expression.

        The function assumes the file path contains a date in 'YYYY/MM/DD' format.
    
        Returns:
        - datetime.date: The extracted date.

        Raises:
        - ValueError: If the date is not found in the file path.
        """
        normalised_filepath = os.path.normpath(filepath)
        date_pattern = r'(\d{4})[/\\](\d{2})[/\\](\d{2})'
        match = re.search(date_pattern, normalised_filepath)

        if match:
            year, month, day = map(int, match.groups())
            return datetime.date(year, month, day)
        else:
            raise ValueError(f"Date not found in file path: {filepath}")

    @staticmethod
    def _get_dataframe(filepath) -> pd.DataFrame:
        df = Processor.unpickle(filepath)
        if not df.empty:
            return df
        else:
            return pd.DataFrame()

    def prepare_dataframe(self) -> bool:
        """
        Prepares the DataFrame for analysis by filtering based on specified criteria.

        This includes checking the contents and converting 'Datetime' to datetime objects.

        Returns:
        - tuple: A boolean indicating if the DataFrame is prepared.
        """
        # Retrieve the DataFrame contained in the file at the location filepath
        self.df = Postprocessor._get_dataframe(self.filepath)
        # Check if DataFrame contains data and required columns
        self.is_df_prepared = Processor.check_df(self.filepath, self.df)
        if self.is_df_prepared:
            # Format datetime string as a dattime object
            self.df['Datetime'] = pd.to_datetime(self.df['Datetime'], format='%Y%m%d%H%M')
            
            # # Sort out bad measurements from "clear-sky" or "clear-ish sky" (correct for
            # # habit of OBR extraction code not updating cloud phase for clear sky measurements)
            # reduced_fields = Processor._get_reduced_fields()
            ### UNDER CONSTRUCTION
            return
        else:
            # Create a new DataFrame with the datetime value
            datetime_value = pd.to_datetime(Postprocessor.extract_date_from_filepath(self.filepath), format='%Y%m%d%H%M')
            dummy_df = pd.DataFrame({'Datetime': [datetime_value]})

            # Concatenate the new row to the existing DataFrame
            self.df = pd.concat([self.df, dummy_df], ignore_index=True)
            return

    @staticmethod
    def set_as_invalid():
        # Return a dictionary with a specific structure or flag to indicate invalid data
        return {"invalid": True}


    def get_outgoing_longwave_radiation(self) -> Dict[str, Union[float, int]]:
        """
        Calculates OLR values for all CloudPhase conditions from the DataFrame.

        Returns:
        - dict: Dictionary with OLR values for each CloudPhase condition.
        """
        # Initialize an empty dictionary to store values.
        olr_values = {}

        # Iterate over each category and store values
        for phase, name in self.cloud_phase_names.items():
            if self.is_df_prepared:
                # For all rows with CloudPhase1 == phase, create sub_df with values of cloud phase, cloud fraction and spectral channels 
                filtered_df = self.df_binned[self.df_binned['CloudPhase1'] == phase]
                olr = filtered_df['OLR']
                olr_values[name] = olr
            else:
                olr_values[name] = -1

        return olr_values
    

    def get_phase_fraction(self) -> Dict[str, Union[float, int]]:
        """
        Calculates the fraction of measurements for each CloudPhase in the DataFrame.

        Returns:
        - dict: A dictionary with CloudPhase names as keys and their corresponding fractions as values.
        """
        # Initialize an empty dictionary to store values.
        phase_fractions = {}

        if self.is_df_prepared:
            # Pivot the DataFrame once to get the individual counts of each Cloud Phase
            pivot_df = self.df_binned.pivot_table(index=self.df_binned['Datetime'].dt.date, columns='CloudPhase1', aggfunc='size', fill_value=0)

            # Calculate total number of measurements for the entire day
            total_measurements = pivot_df.sum(axis=1).sum()

            # Iterate over each CloudPhase category and calculate fractions
            for phase, name in self.cloud_phase_names.items():
                phase_count = pivot_df.get(phase, pd.Series(dtype="int32")).sum()  # Use .get() to avoid KeyError
                phase_fractions[name] = 0 if total_measurements == 0 else np.round(phase_count / total_measurements, 3)
        else:
            # DataFrame is not prepared, assign default value for missing data
            for name in self.cloud_phase_names.values():
                phase_fractions[name] = -1

        return phase_fractions


    def process_target_variables(self, target_variables, data_dict) -> None:
        """
        Processes each target variable and appends the results to the data dictionary.

        Parameters:
        - target_variables (list): List of target variables to process.
        - data_dict (dict): Dictionary to store the results.
        """
        for var in target_variables:
            if var == 'OLR':
                values = self.get_outgoing_longwave_radiation()
            elif var == 'Phase Fraction':
                values = self.get_phase_fraction()
            else:
                logger.info(f"Target variable not recognised: {var}")
                values = Postprocessor.set_as_invalid()

            for key, value in values.items():
                print(key, value)
                data_dict[var][key].append(value)

        return None
    
    @staticmethod
    def save_results(data_dict, dates, datapath) -> None:
        """
        Saves the processed results into CSV files.

        Parameters:
        - data_dict (defaultdict): Nested dictionary containing the results for each target variable.
        - dates (list): List of dates corresponding to each entry in the data dictionary.
        - datapath (str): Path to save the CSV files.
        """
        for var, results in data_dict.items():
            # Check if variable dictionary contains invalid values
            if ("invalid" in results) and results["invalid"]:
                # Skip saving for unrecognised variables
                continue
            
            # Store dates to first column of DataFrame
            df_to_save = pd.DataFrame({'Date': pd.to_datetime(dates)})
            
            # For each key in the inner dictionary, create a new column in the DataFrame
            for key, value in results.items():
                df_to_save[key] = value
            
            # Build filename and save to CSV
            filename = f"daily_{var.lower().replace(' ', '_')}.csv"
            df_to_save.to_csv(os.path.join(datapath, filename), sep="\t", index=False)

        return None