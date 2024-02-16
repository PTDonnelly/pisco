import gzip
import logging
import os
import pandas as pd
from typing import List, Optional

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
    

    def check_l1c_l2_data_exist(self):
        
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
    def unpickle(file):
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
    def _get_reduced_fields() -> List[str]:
        reduced_fields = [
            "Datetime", "Latitude", 'Longitude', "SatelliteZenithAngle", "DayNightQualifier",
            "Pressure1", " Temperature/dry-bulbTemperature1", "CloudAmountInSegment1", "CloudPhase1"]
        return reduced_fields


    def reduce_fields(self, merged_df: pd.DataFrame):
        # Keep only columns containing variables present in reduced_fields and spectral channels
        reduced_fields = Processor._get_reduced_fields()
        spectrum_columns = [col for col in merged_df if "Spectrum" in col]
        return merged_df.filter(reduced_fields + spectrum_columns)
    
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
        
        logger.info(f"DataFrame processed: {filepath}")
        return True


    def filter_observations(self, df, maximum_zenith_angle=5):
        """
        Prepares the dataframe by converting 'Datetime' to pandas datetime objects,
        removing missing data, and filtering for SatelliteZenithAngle less than 5 degrees.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame containing satellite data.
        maximum_zenith_angle (int): Maximum satellite zenith angle considered (<5 degrees is considered nadir-viewing)

        Returns:
        pd.DataFrame: Filtered and processed DataFrame.
        """
        # Check if DataFrame contains data and required columns
        df_good = Processor.check_df(self.output_path, df)
        
        if not df_good:
            # If Dataframe is missing values or columns, return empty dataframe
            return pd.DataFrame()
        else:
            # Keep rows where 'SatelliteZenithAngle' is less than the specified maximum zenith angle (default = 5 degrees, considered to be nadir)
            include_nadir = df['SatelliteZenithAngle'] < maximum_zenith_angle

            # # Discard measurements where clouds are in liquid phase
            # exclude_liquid = df['CloudPhase1'] != 2
            
            # # Discard measurements where clouds are in mixed phase
            # exclude_mixed = df['CloudPhase1'] != 3

            # Combine all conditions using the bitwise AND operator
            combined_conditions = include_nadir

            # Filter the DataFrame based on the combined conditions
            filtered_df = df[combined_conditions]

            # Check that DataFrame still contains data after filtering
            if filtered_df.empty:
                logger.info(f"No data remains after filtering: {self.output_path}")
                return pd.DataFrame()
            else:
                return filtered_df


    def merge_datasets(self) -> None:
        # Latitude and longitude values are rounded to 4 decimal places.
        self.df_l1c[['Latitude', 'Longitude']] = self.df_l1c[['Latitude', 'Longitude']].round(4)
        self.df_l2[['Latitude', 'Longitude']] = self.df_l2[['Latitude', 'Longitude']].round(4)

        # Merge two DataFrames based on spatial and temporal parameters
        return pd.merge(self.df_l1c, self.df_l2, on=["Datetime", "Latitude", 'Longitude', "SatelliteZenithAngle"], how='inner')


    def _create_merged_datapath(self):
        # Create the output directory if it doesn't exist
        os.makedirs(self.datapath_merged, exist_ok=True)
        self.output_path = os.path.join(self.datapath_merged, "spectra_and_cloud_products")
        return


    def combine_datasets(self) -> None:
        self._create_merged_datapath()
        
        # Merge two DataFrames based on space-time co-ordinates
        merged_df = self.merge_datasets()

        # Filter merged dataset to throw away unwanted or bad measurements
        filtered_df = self.filter_observations(merged_df)

        # Reduce dataset to specified parameters
        self.df = self.reduce_fields(filtered_df)
        return
    

    def _delete_intermediate_file(self, filepath):
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
            
            # Output information on the final DataFrame
            logger.info(self.df.info())
            logger.info(self.df.head())
            logger.info(f"Saved merged products to: {output_file}")

        except OSError as e:
            logger.error(f"Error saving file: {e}")

        # Delete Preprocessor files
        if (delete_intermediate_files is None) and self.delete_intermediate_files:
            # If boolean flag is not manually passed, default to the boolean flag in config.delete_intermediate_files
                self._delete_intermediate_file(self.datafile_l1c)
                self._delete_intermediate_file(self.datafile_l2)

        return