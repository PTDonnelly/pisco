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
    def __init__(self, ex: Extractor):
        self.datapath_l1c = f"{ex.config.datapath}{ex.config.satellite_identifier}/l1c/{ex.year}/{ex.month}/{ex.day}/"
        self.datapath_l2 = f"{ex.config.datapath}{ex.config.satellite_identifier}/l2/{ex.year}/{ex.month}/{ex.day}/"
        self.datapath_merged = f"{ex.config.datapath}{ex.config.satellite_identifier}/{ex.year}/{ex.month}/{ex.day}/"
        self.delete_intermediate_files = ex.config.delete_intermediate_files
        self.output_path: str = None
        self.df_l1c: object = None
        self.df_l2: object = None
        self.df: object = None


    def _get_intermediate_analysis_data_paths(self) -> None:
        """
        Defines the paths to the intermediate analysis data files.
        """
        self.datafile_l1c = f"{self.datapath_l1c}extracted_spectra.pkl.gz"
        self.datafile_l2 = f"{self.datapath_l2}cloud_products.pkl.gz"
        return
    

    def check_l1c_l2_data_exist(self):
        
        self._get_intermediate_analysis_data_paths()
        
        # Check if L1C and/or L2 data files exist
        if not os.path.exists(self.datafile_l1c) and not os.path.exists(self.datafile_l2):
            logging.info('Neither L1C nor L2 data files exist. Nothing to correlate.')
            return False
        elif not os.path.exists(self.datafile_l1c):
            logging.info('L1C data files do not exist. Cannot correlate.')
            return False
        elif not os.path.exists(self.datafile_l2):
            logging.info('L2 data files do not exist. Cannot correlate.')
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
        logging.info("Loading L1C spectra and L2 cloud products:")
        self.df_l1c = Processor.unpickle(self.datafile_l1c)
        self.df_l2 = Processor.unpickle(self.datafile_l2)
        return
    

    def _check_headers(self):
        required_headers = ['Latitude', 'Longitude', 'Datetime', 'Local Time']
        missing_headers_l1c = [header for header in required_headers if header not in self.df_l1c.columns]
        missing_headers_l2 = [header for header in required_headers if header not in self.df_l2.columns]
        if missing_headers_l1c or missing_headers_l2:
            raise ValueError(f"Missing required headers in df_l1c: {missing_headers_l1c} or df_l2: {missing_headers_l2}")


    def correlate_datasets(self) -> None:
        """
        Create a single DataFrame for all contemporaneous observations
        """
        # Check that latitude, longitude, datetime, and local time are present in both file headers 
        self._check_headers()

        # Latitude and longitude values are rounded to 4 decimal places.
        self.df_l1c[['Latitude', 'Longitude']] = self.df_l1c[['Latitude', 'Longitude']].round(4)
        self.df_l2[['Latitude', 'Longitude']] = self.df_l2[['Latitude', 'Longitude']].round(4)
        return

    @staticmethod
    def _get_reduced_fields() -> List[str]:
        reduced_fields = [
            "Datetime", "Latitude", 'Longitude', "SatelliteZenithAngle", "DayNightQualifier",
            "Pressure1", "TemperatureOrDryBulbTemperature1", "CloudAmountInSegment1",
            "CloudPhase1", "CloudPhase2", "CloudPhase3"]
        
        return reduced_fields


    def reduce_fields(self, merged_df: pd.DataFrame):
        # Keep only columns containing variables present in reduced_fields and spectral channels
        reduced_fields = Processor._get_reduced_fields()
        spectrum_columns = [col for col in merged_df if "Spectrum" in col]
        return merged_df.filter(reduced_fields + spectrum_columns)
    
    @staticmethod
    def check_df(filepath: str, df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
        # Ensure the dataframe is not empty
        if df.empty:
            logging.info(f"DataFrame empty: {filepath}")      
            return False     

        # Check for the presence of all required columns 
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.info(f"Missing column(s) in DataFrame: {filepath}\n{', '.join(missing_columns)}")
                return False
        logging.info(f"DataFrame processed: {filepath}")
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
        # Check if DataFrame contains data and required columns are present
        filtering_columns = ['CloudPhase1', 'CloudPhase2', 'CloudPhase3', 'SatelliteZenithAngle']
        df_good = Processor.check_df(self.output_path, df, filtering_columns)
        
        if not df_good:
            # If Dataframe is missing values or columns, return empty dataframe
            return pd.DataFrame()
        else:
            # # Keep rows where 'CloudPhase1' is not -1 (-1 is a bad measurement indicator, throw these measurements)
            # condition_1 = df['CloudPhase1'] != -1
            # # Keep rows where 'CloudPhase2' is -1 (throw measurements with multiple cloud phases)
            # condition_2 = df['CloudPhase2'] == -1
            # # Keep rows where 'CloudPhase3' is -1 (throw measurements with multiple cloud phases)
            # condition_3 = df['CloudPhase3'] == -1
            # # Keep rows where 'SatelliteZenithAngle' is less than the specified maximum zenith angle (default = 5 degrees, considered to be nadir)
            # condition_4 = df['SatelliteZenithAngle'] < maximum_zenith_angle

            # # Combine all conditions using the bitwise AND operator
            # combined_conditions = condition_1 & condition_2 & condition_3 & condition_4

            # Keep rows where 'CloudPhase1' is not -1 (-1 is a bad measurement indicator, throw these measurements)

            # Keep rows where 'SatelliteZenithAngle' is less than the specified maximum zenith angle (default = 5 degrees, considered to be nadir)
            condition_4 = df['SatelliteZenithAngle'] < maximum_zenith_angle

            # Combine all conditions using the bitwise AND operator
            combined_conditions = condition_4

            # Filter the DataFrame based on the combined conditions
            filtered_df = df[combined_conditions]

            # Check that DataFrame still contains data after filtering
            if filtered_df.empty:
                logging.info(f"No data remains after filtering: {self.output_path}")
                return pd.DataFrame()
            else:
                return filtered_df


    def merge_datasets(self) -> None:
        # Merge two DataFrames based on latitude, longitude and datetime,
        # rows from df_l1c that do not have a corresponding row in df_l2 are dropped.
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
            logging.info(f"Deleted intermediate file: {filepath}")
        except OSError as e:
            logging.error(f"Error deleting file {filepath}: {e}")
        return

    def save_merged_products(self, delete_intermediate_files: Optional[bool]=None) -> None:
        if not self.df.empty:
            try:
                # Compress and save using gzip
                with gzip.open(f"{self.output_path}.pkl.gz", 'wb') as f:
                    pickle.dump(self.df, f)

                self.df.to_csv(f"{self.output_path}.csv", sep='\t', index=False)
                
                logging.info(f"Saved merged products to: {self.output_path}.pkl.gz")

            except OSError as e:
                logging.error(f"Error saving file: {e}")

        # Delete Preprocessor files
        if (delete_intermediate_files is None) and self.delete_intermediate_files:
            # If boolean flag is not manually passed, default to the boolean flag in config.delete_intermediate_files
                self._delete_intermediate_file(self.datafile_l1c)
                self._delete_intermediate_file(self.datafile_l2)

        return