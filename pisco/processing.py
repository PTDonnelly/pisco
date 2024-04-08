import gzip
import logging
import numpy as np
import os
import pandas as pd
from typing import List, Optional, Union

import pickle

from pisco import Extractor

# Obtain a logger for this module
logger = logging.getLogger(__name__)

class Processor:
    """Processes and merges L1C spectra and L2 cloud product data for IASI.

    This class is responsible for managing the data processing pipeline for IASI data, including loading,
    merging, and filtering data from L1C and L2 products based on spatial and temporal parameters. It also
    handles the reduction of dataset fields and the saving of the merged products.

    Attributes:
        datapath_l1c (str): Path to L1C data files.
        datapath_l2 (str): Path to L2 data files.
        datapath_merged (str): Path for saving merged data files.
        delete_intermediate_files (bool): Flag to indicate whether intermediate files should be deleted after processing.
        output_path (str): Path for the output file, determined during processing.
        df_l1c (pd.DataFrame or None): DataFrame containing L1C data.
        df_l2 (pd.DataFrame or None): DataFrame containing L2 data.
        df (pd.DataFrame or None): DataFrame containing the merged and processed L1C and L2 data.

    Methods:
        _get_intermediate_analysis_data_paths(): Sets the paths for intermediate analysis data files.
        check_l1c_l2_data_exist(): Checks the existence of L1C and L2 data files.
        unpickle(file): Uncompresses and loads a DataFrame from a pickled file.
        load_data(): Loads L1C spectra and L2 cloud products data into DataFrames.
        _get_iasi_spectral_grid(): Retrieves the IASI spectral grid from a file.
        get_dataframe_spectral_grid(): Extracts the spectral grid from the DataFrame's columns.
        calculate_olr_from_spectrum(sub_df): Calculates the OLR from spectral data for a given day.
        _get_reduced_fields(): Returns a list of fields to retain in the reduced dataset.
        reduce_fields(merged_df): Reduces the merged DataFrame to specified fields and spectral channels.
        check_df(filepath, df): Validates the DataFrame for data presence and required columns.
        filter_observations(df, maximum_zenith_angle): Filters observations based on zenith angle and cloud phase.
        merge_datasets(): Merges L1C and L2 DataFrames on spatial and temporal parameters.
        _create_merged_datapath(): Creates the output directory for merged products if it does not exist.
        combine_datasets(): Merges, filters, and reduces datasets, and sets the final DataFrame.
        _delete_intermediate_file(filepath): Deletes the specified intermediate file if required.
        save_merged_products(delete_intermediate_files): Saves the merged products to a file and optionally deletes intermediate files.
    """
    def __init__(self, ex: Extractor):
        self.datapath_l1c = os.path.join(ex.config.datapath, ex.config.satellite_identifier,'l1c', ex.year, ex.month, ex.day)
        self.datapath_l2 = os.path.join(ex.config.datapath, ex.config.satellite_identifier,'l2', ex.year, ex.month, ex.day)
        self.datapath_merged = os.path.join(ex.config.datapath, ex.config.satellite_identifier, ex.year, ex.month, ex.day)
        self.delete_intermediate_files = ex.config.delete_intermediate_files
        self.output_path: str = None
        self.df_l1c: object = None
        self.df_l2: object = None
        self.df: object = None


    def _get_intermediate_analysis_data_paths(self) -> None:
        """
        Defines the paths to the intermediate analysis data files.
        """
        self.datafile_l1c = os.path.join(self.datapath_l1c, "extracted_spectra.pkl.gz")
        self.datafile_l2 =  os.path.join(self.datapath_l2, "cloud_products.pkl.gz")
        return
    

    def check_l1c_l2_data_exist(self) -> bool:
        
        self._get_intermediate_analysis_data_paths()
        
        # Check if L1C and/or L2 data files exist
        if not os.path.exists(self.datafile_l1c) and not os.path.exists(self.datafile_l2):
            logger.info('Neither L1C nor L2 data files exist. Nothing to correlate.')
            return False
        elif not os.path.exists(self.datafile_l1c):
            logger.info('L1C data files do not exist. Cannot correlate.')
            return False
        elif not os.path.exists(self.datafile_l2):
            logger.info('L2 data files do not exist. Cannot correlate.')
            return False
        else:
            return True
    
    @staticmethod
    def unpickle(file) -> pd.DataFrame:
        with gzip.open(file, 'rb') as f:
            df = pickle.load(f)
        return df
    

    def load_data(self) -> None:
        """
        Uncompresses two pickled DataFrames loaded from the intermediate analysis data files.
        
        """
        logger.info("Loading L1C spectra and L2 cloud products:")
        self.df_l1c = Processor.unpickle(self.datafile_l1c)
        self.df_l2 = Processor.unpickle(self.datafile_l2)
        return
    
    @staticmethod
    def check_df(filepath: str, df: pd.DataFrame = None) -> bool:
        # Ensure the dataframe is not empty
        if df.empty:
            logger.info(f"DataFrame empty: {filepath}")      
            return False

        # Check for the presence of all required columns
        required_columns = Processor._get_reduced_fields()
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.info(f"Missing column(s) in DataFrame: {filepath}\n{', '.join(missing_columns)}")
                return False
        
        logger.info(f"DataFrame checked: {filepath}")
        return True
            
    @staticmethod
    def _get_iasi_spectral_grid():
        spectral_grid = np.loadtxt("./inputs/iasi_spectral_grid.txt")
        channel_ids = spectral_grid[:, 0]
        wavenumber_grid = spectral_grid[:, 1]
        return channel_ids, wavenumber_grid
    

    def get_dataframe_spectral_grid(self) -> List[float]:
        # Get the full IASI spectral grid
        _, wavenumber_grid = self._get_iasi_spectral_grid()
        
        # Extract the numbers from the column names
        spectral_channels = self.df[[col for col in self.df.columns if 'Spectrum' in col]]

        print(spectral_channels)
        
        channel_positions = spectral_channels.columns.str.split().str[-1].astype(int)

        print(channel_positions)
        input()

        # Extract the wavenumbers corresponding to the channel positions
        extracted_wavenumbers = [wavenumber_grid[position-1] for position in channel_positions]
        return extracted_wavenumbers
                

    def _create_merged_datapath(self):
        # Create the output directory if it doesn't exist
        os.makedirs(self.datapath_merged, exist_ok=True)
        self.output_path = os.path.join(self.datapath_merged, "spectra_and_cloud_products")
        return


    def merge_datasets(self) -> None:
        # Latitude and longitude values are rounded to 4 decimal places.
        self.df_l1c[['Latitude', 'Longitude']] = self.df_l1c[['Latitude', 'Longitude']].round(4)
        self.df_l2[['Latitude', 'Longitude']] = self.df_l2[['Latitude', 'Longitude']].round(4)

        # Ensure both Datetime columns are datetime objects
        self.df_l1c['Datetime'] = pd.to_datetime(self.df_l1c['Datetime'])
        self.df_l2['Datetime'] = pd.to_datetime(self.df_l2['Datetime'])

        # Merge two DataFrames based on spatial and temporal parameters
        return pd.merge(self.df_l1c, self.df_l2, on=["Datetime", "Latitude", 'Longitude', 'DayNightQualifier'], how='inner')

    @staticmethod
    def build_filter_conditions(df: pd.DataFrame, maximum_zenith_angle: int=5):
        """
        Defines and combines multiple filtering conditions for creation of
        final merged dataset.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing satellite data.
        maximum_zenith_angle (int): Maximum satellite zenith angle considered (<5 degrees is considered nadir-viewing)
        """
        # Keep rows where 'SatelliteZenithAngle' is less than the specified maximum zenith angle (default = 5 degrees, considered to be nadir)
        include_nadir = df['SatelliteZenithAngle'] < maximum_zenith_angle

        
        # Combine all conditions using the bitwise AND operator (e.g. condition1 & condition2 etc.)
        combined_conditions = include_nadir
        return combined_conditions
    

    def filter_observations(self, df):
        """Prepares the dataframe by converting 'Datetime' to pandas datetime objects,
        removing missing data, and filtering based on conditions.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame containing satellite data.

        Returns:
        pd.DataFrame: Filtered and processed DataFrame.
        """
        # Check if DataFrame contains data and required columns
        df_good = Processor.check_df(self.output_path, df)
        
        if not df_good:
            # If Dataframe is missing values or columns, return empty dataframe
            return pd.DataFrame()
        else:
            # Build filtering conditions for data set
            combined_conditions = Processor.build_filter_conditions(df)

            # Filter the DataFrame based on the combined conditions
            filtered_df = df[combined_conditions]

            # Check that DataFrame still contains data after filtering
            if filtered_df.empty:
                logger.info(f"No data remains after filtering: {self.output_path}")
                return pd.DataFrame()
            else:
                return filtered_df

    @staticmethod
    def _get_reduced_fields() -> List[str]:
        reduced_fields = [
            "Datetime", "Latitude", 'Longitude', "SatelliteZenithAngle", "DayNightQualifier",
            'CloudTopPressure1', "CloudTopTemperature1", "CloudAmountInSegment1", "CloudPhase1"]
        return reduced_fields


    def reduce_fields(self, merged_df: pd.DataFrame):
        # Keep only columns containing variables present in reduced_fields and spectral channels
        reduced_fields = Processor._get_reduced_fields()
        spectrum_columns = [col for col in merged_df if "Spectrum" in col]
        return merged_df.filter(reduced_fields + spectrum_columns)


    def calculate_olr_and_error(self, row, wavenumbers):
        """Function to calculate OLR and its error for a single spectrum row"""
        radiance = row[[col for col in self.df.columns if 'Spectrum' in col]].values
        radiance_error = np.std(radiance) # Adapt later to utilise IASI noise profile
        radiance_errors = np.full(radiance.shape, radiance_error)

        integrated_spectrum = np.trapz(radiance, wavenumbers)
        integrated_error = np.sqrt(np.trapz(radiance_errors ** 2, wavenumbers))

        cloud_fraction = row['CloudAmountInSegment1'] / 100  # Convert percentage to fraction
        weighted_integrated_spectrum = integrated_spectrum * cloud_fraction
        weighted_integrated_error = integrated_error * cloud_fraction

        return pd.Series((weighted_integrated_spectrum, weighted_integrated_error))
    

    def integrate_spectrum_to_olr(self) -> None:
        """
        Calculates OLR and its error for each spectrum in the DataFrame, adds these as new columns,
        and removes the original spectral radiance columns.
        """

        # Report to the logger
        logger.info("Integrating spectra to OLR")

        # Retrieve IASI spectral grid and radiance from the DataFrame
        wavenumbers = self.get_dataframe_spectral_grid()

        # Apply the function to each row, assuming spectral and error data are correctly aligned
        self.df[['OLR', 'OLR_Error']] = self.df.apply(self.calculate_olr_and_error, wavenumbers=wavenumbers, axis=1)

        # Remove the original spectral radiance and error columns
        self.df.drop(columns=[col for col in self.df.columns if 'Spectrum' in col], inplace=True)
        return
    

    def downsample_spatial_grid(self):
        """
        Performs spatial binning of measurements onto a 1x1 degree lat-lon grid and averages
        the measurements within each bin. A new Datetime column is added back, containing
        just the date part of the original datetime objects.
        """
        # Convert 'Datetime' strings to datetime objects
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'], format='%Y%m%d%H%M')
        # Extract just the date part from the 'Datetime' column
        self.df['Date'] = self.df['Datetime'].dt.date

        # Round latitude and longitude to nearest whole number to create grid bins
        self.df['Latitude_binned'] = self.df['Latitude'].round().astype(int)
        self.df['Longitude_binned'] = self.df['Longitude'].round().astype(int)

        # Group by the new lat-lon bins and 'Date', then calculate mean of measurements for each bin
        # Exclude original Latitude, Longitude, and Datetime columns from the mean calculation
        grouped = self.df.groupby(['Latitude_binned', 'Longitude_binned', 'Date']).mean()

        # Reset index to turn grouped DataFrame back into a format that resembles the original df
        df_binned = grouped.reset_index()

        # Drop the original Latitude, Longitude, and Datetime columns from the binned df
        df_binned.drop(columns=['Latitude', 'Longitude', 'Datetime'], errors='ignore', inplace=True)

        # Rename the binned latitude and longitude columns to 'Latitude' and 'Longitude'
        df_binned.rename(columns={'Latitude_binned': 'Latitude', 'Longitude_binned': 'Longitude'}, inplace=True)

        # Replace the original DataFrame with the binned version
        self.df = df_binned

        # Ensure the DataFrame is sorted by Latitude and Longitude for readability and consistency
        self.df = self.df.sort_values(by=['Latitude', 'Longitude']).reset_index(drop=True)
        
        return


    def combine_datasets(self) -> None:
        self._create_merged_datapath()
        
        # Merge two DataFrames based on space-time co-ordinates
        merged_df = self.merge_datasets()

        # Filter merged dataset to throw away unwanted or bad measurements
        filtered_df = self.filter_observations(merged_df)

        # Reduce dataset to specified parameters
        self.df = self.reduce_fields(filtered_df)

        # Integrate spectra with wavelength to produce a single OLR value
        self.integrate_spectrum_to_olr()

        # Downsample measurements on to 1x1 degree spatial grid
        self.downsample_spatial_grid()
        return
    
    @staticmethod
    def _delete_intermediate_file(filepath):
        try:
            os.remove(filepath)
            logger.info(f"Deleted intermediate file: {filepath}")
        except OSError as e:
            logger.error(f"Error deleting file {filepath}: {e}")
        return


    def save_merged_products(self, delete_intermediate_files: Optional[bool]=None) -> None:
        try:
            # Split the intermediate file path into the root and extension, and give new extension
            file_root, _ = os.path.splitext(self.output_path)
            output_file = file_root + ".pkl.gz"

            # Compress and save using gzip
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(self.df, f)

            self.df.to_csv(f"{file_root}.csv", sep='\t', index=False)
            
            # Output information on the final DataFrame
            logger.info(self.df.info())
            logger.info(self.df.head())
            logger.info(f"Saved merged products to: {output_file}")

        except OSError as e:
            logger.error(f"Error saving file: {e}")

        # Delete Preprocessor files
        if (delete_intermediate_files is None) and self.delete_intermediate_files:
            # If boolean flag is not manually passed, default to the boolean flag in config.delete_intermediate_files
                Processor._delete_intermediate_file(self.datafile_l1c)
                Processor._delete_intermediate_file(self.datafile_l2)

        return