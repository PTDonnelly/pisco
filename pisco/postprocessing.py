import numpy as np
import datetime
import os.path as ospath
import re
import pandas as pd
from typing import List

from pisco import Processor

class Postprocessor:
    def __init__(self, df: pd.DataFrame, filepath: str):
        self.df = df
        self.filepath = filepath

    def prepare_dataframe(self):
        """
        Prepares the DataFrame for analysis by filtering based on specified criteria.

        This includes checking the contents and converting 'Datetime' to datetime objects.

        Returns:
        - tuple: A boolean indicating if the DataFrame is prepared.
        """
        required_columns = ['CloudPhase1', 'SatelliteZenithAngle', 'Datetime']
        if not Processor.check_df(self.filepath, self.df, required_columns):
            # Report if Dataframe is missing values or columns
            return False
        else:
            # Proceed with DataFrame manipulations if all required columns are present
            self.df['Datetime'] = pd.to_datetime(self.df['Datetime'], format='%Y%m%d%H%M')
            return True

    @staticmethod
    def _get_iasi_spectral_grid():
        spectral_grid = np.loadtxt("./inputs/iasi_spectral_grid.txt")
        channel_ids = spectral_grid[:, 0]
        wavenumber_grid = spectral_grid[:, 1]
        return channel_ids, wavenumber_grid

    def get_dataframe_spectral_grid(self) -> List[float]:
        # Get the full IASI spectral grid
        _, wavenumber_grid = Postprocessor._get_iasi_spectral_grid()
        # Extract the numbers from the column names
        spectral_channels = self.df[[col for col in self.df.columns if 'Spectrum' in col]]
        channel_positions = spectral_channels.columns.str.split().str[-1].astype(int)
        # Extract the wavenumbers corresponding to the channel positions
        extracted_wavenumbers = [wavenumber_grid[position] for position in channel_positions]
        return extracted_wavenumbers

    def calculate_olr_from_spectrum(self, sub_df):
        """
        Calculates the Outgoing Longwave Radiation (OLR) from spectral data.

        Returns:
        - float: The calculated OLR value.
        """
        # Retrieve IASI spectral grid and radiance from the DataFrame
        wavenumbers = self.get_dataframe_spectral_grid()
        radiance_df = sub_df[[col for col in sub_df.columns if 'Spectrum' in col]]

        # Convert wavenumbers to wavelengths in meters
        wavelengths = 1e-2 / np.array(wavenumbers)  # Conversion from cm^-1 to m
        # Convert radiance to SI units: W/m^2/sr/m
        radiance_si = radiance_df.values * 1e-3  # Convert from mW to W
        
        # Integrate the radiance over the wavelength and sum integral elements
        olr_integrals = np.trapz(radiance_si, wavelengths, axis=1)
        return np.sum(olr_integrals)

    def get_outgoing_longwave_radiation(self):
        """
        Calculates OLR values for icy and clear-sky conditions from the DataFrame.

        Returns:
        - tuple: OLR values for clear-sky and icy conditions.
        """
        icy_olr = self.calculate_olr_from_spectrum(self.df[self.df['CloudPhase1'] == 2])
        clear_olr = self.calculate_olr_from_spectrum(self.df[self.df['CloudPhase1'] == 4])
        
        return clear_olr, icy_olr
    
    def get_ice_fraction(self):
        """
        Calculates the fraction of icy cloud measurements in the DataFrame.

        Returns:
        - float: The fraction of measurements flagged as icy.
        """
        # Pivot the DataFrame to get the individual counts of each Cloud Phase per Datetime
        pivot_df = self.df.pivot_table(index=self.df['Datetime'].dt.date, columns='CloudPhase1', aggfunc='size', fill_value=0)

        # Calculate total number of measurements for the entire day
        total_measurements = pivot_df.sum(axis=1).values
        
        # Calculate proportion of measurements for the entire day flagged as "icy"
        ice_count = pivot_df.get(2, 0).sum()

        return np.round(ice_count / total_measurements, 3) if total_measurements.sum() > 0 else 0

    def process_target_variables(self, target_variables, data_dict):
        """
        Processes each target variable and appends the results to the data dictionary.

        Parameters:
        - target_variables (list): List of target variables to process.
        - data_dict (dict): Dictionary to store the results.
        """
        for var in target_variables:
            if var == 'OLR':
                result = self.get_outgoing_longwave_radiation()
            elif var == 'Ice Fraction':
                result = self.get_ice_fraction()
            else:
                print(f"Target variable not recognised: {var}")
                continue
            
            data_dict[var].append(result)

    @staticmethod
    def append_bad_values(target_variables, data_dict):
        """
        Appends bad values for each target variable in the data dictionary.

        Parameters:
        - target_variables (list): List of target variables.
        - data_dict (dict): Dictionary to store the results.
        """
        for var in target_variables:
            data_dict[var].append([-1, -1])  # Assuming [-1, -1] represents bad or missing data


    def extract_date_from_filepath(self):
        """
        Extracts the date from a file path using a regular expression.

        The function assumes the file path contains a date in 'YYYY/MM/DD' format.
    
        Returns:
        - datetime.date: The extracted date.

        Raises:
        - ValueError: If the date is not found in the file path.
        """
        normalised_filepath = ospath.normpath(self.filepath)
        date_pattern = r'(\d{4})[/\\](\d{2})[/\\](\d{2})'
        match = re.search(date_pattern, normalised_filepath)

        if match:
            year, month, day = map(int, match.groups())
            return datetime.date(year, month, day)
        else:
            raise ValueError(f"Date not found in file path: {self.filepath}")
        
    @staticmethod
    def save_results(data_dict, dates, datapath):
        """
        Saves the processed results into CSV files.

        Parameters:
        - data_dict (dict): Dictionary containing the results for each target variable.
        - dates (list): List of dates corresponding to each entry in the data dictionary.
        - datapath (str): Path to save the CSV files.
        - target_variables (list): List of target variables.
        """
        for var, results in data_dict.items():
            df_to_save = pd.DataFrame({'Date': pd.to_datetime(dates)})
            if var == 'Ice Fraction':
                df_to_save[var] = [result if not isinstance(result, (list, np.ndarray)) else result[0] for result in results]
            elif var == 'OLR':
                clear_sky, icy = zip(*results)
                df_to_save[f'{var} Clear-Sky'] = clear_sky
                df_to_save[f'{var} Icy'] = icy

            df_to_save.to_csv(ospath.join(datapath, f"daily_{var.lower().replace(' ', '_')}.csv"), index=False)