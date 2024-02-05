import gzip
import os
import pandas as pd
from typing import List

import pickle

from pisco import Extractor, Plotter

class Processor:
    def __init__(self, ex: Extractor):
        self.cloud_phase: int = ex.config.cloud_phase
        self.datapath_l1c = f"{ex.config.datapath}{ex.config.satellite_identifier}/l1c/{ex.year}/{ex.month}/{ex.day}/"
        self.datapath_l2 = f"{ex.config.datapath}{ex.config.satellite_identifier}/l2/{ex.year}/{ex.month}/{ex.day}/"
        self.datapath_merged = f"{ex.config.datapath}{ex.config.satellite_identifier}/{ex.year}/{ex.month}/{ex.day}/"
        self.df_l1c: object = None
        self.df_l2: object = None

    def _get_intermediate_analysis_data_paths(self) -> None:
        """
        Defines the paths to the intermediate analysis data files.
        """
        self.datafile_l1c = f"{self.datapath_l1c}extracted_spectra.csv"
        self.datafile_l2 = f"{self.datapath_l2}cloud_products.csv"
        return
    
    def check_l1c_l2_data_exist(self):
        
        self._get_intermediate_analysis_data_paths()
        
        # Check if L1C and/or L2 data files exist
        if not os.path.exists(self.datafile_l1c) and not os.path.exists(self.datafile_l2):
            print('Neither L1C nor L2 data files exist. Nothing to correlate.')
            return False
        elif not os.path.exists(self.datafile_l1c):
            print('L1C data files do not exist. Cannot correlate.')
            return False
        elif not os.path.exists(self.datafile_l2):
            print('L2 data files do not exist. Cannot correlate.')
            return False
        else:
            return True
        
    def load_data(self) -> None:
        """
        Opens two DataFrames loaded from the intermediate analysis data files.
        
        """
        # Open csv files
        print("\nLoading L1C spectra and L2 cloud products:")
        self.df_l1c = pd.read_csv(self.datafile_l1c)
        self.df_l2 = pd.read_csv(self.datafile_l2)
        return
    

    def _check_headers(self):
        required_headers = ['Latitude', 'Longitude', 'Datetime', 'Local Time']
        missing_headers_l1c = [header for header in required_headers if header not in self.df_l1c.columns]
        missing_headers_l2 = [header for header in required_headers if header not in self.df_l2.columns]
        if missing_headers_l1c or missing_headers_l2:
            raise ValueError(f"Missing required headers in df_l1c: {missing_headers_l1c} or df_l2: {missing_headers_l2}")
        
    def correlate_measurements(self) -> None:
        """
        Create a single DataFrame for all contemporaneous observations
        """
        # Check that latitude, longitude, datetime, and local time are present in both file headers 
        self._check_headers()

        # Latitude and longitude values are rounded to 4 decimal places.
        self.df_l1c[['Latitude', 'Longitude']] = self.df_l1c[['Latitude', 'Longitude']].round(4)
        self.df_l2[['Latitude', 'Longitude']] = self.df_l2[['Latitude', 'Longitude']].round(4)
        return
    

    def _delete_intermediate_analysis_data(self) -> None:
        """
        Delete the intermediate analysis data files used for correlating spectra and clouds.
        """
        os.remove(self.datafile_l1c)
        os.remove(self.datafile_l2)
        return
    
    def _save_merged_products(self, reduced_df: pd.DataFrame, delete_obr_files: bool = False) -> None:
        print(f"Saving compressed spectra to {output_path}")
        
        # Compress and save using gzip
        with gzip.open(output_path, 'wb') as f:
            pickle.dump(reduced_df, f)

        if delete_obr_files == True:
            # Delete original csv files
            self._delete_intermediate_analysis_data()

    
    def filter_observations(self, output_path, df, maximum_zenith_angle=5):
        """
        Prepares the dataframe by converting 'Datetime' to pandas datetime objects,
        removing missing data, and filtering for SatelliteZenithAngle less than 5 degrees.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame containing satellite data.
        maximum_zenith_angle (int): Maximum satellite zenith angle considered (<5 degrees is considered nadir-viewing)

        Returns:
        pd.DataFrame: Filtered and processed DataFrame.
        """
        required_columns = ['CloudPhase1', 'SatelliteZenithAngle', 'Datetime']
        if Plotter.check_df(df, required_columns):
            # If all required columns are present
            # Drop rows with bad cloud phase measurement
            df = df[df['CloudPhase1'] != -1]
            # Drop rows with multiple cloudphase measurements
            
            # Drop measurements beyond specified zenih angle (default = 5 degrees, considered to be nadir)
            df = df[df['SatelliteZenithAngle'] < maximum_zenith_angle]

            # Check that DataFrame still contains data after filtering
            if not df.empty:
                return True, df
            else:
                print(f"No data remains after filtering: {output_path}")
                return False, None
        else:
            return False, None
    

    @staticmethod
    def _get_reduced_fields() -> List[str]:
        reduced_fields = ["Datetime", "Latitude", 'Longitude', "SatelliteZenithAngle", "DayNightQualifier", "CloudPhase1"]
        return reduced_fields
    
    def reduce_fields(self) -> None:
        # Create the output directory if it doesn't exist
        os.makedirs(self.datapath_merged, exist_ok=True)
        output_path = os.path.join(self.datapath_merged, "spectra_and_cloud_products.pkl.gz")

        # Merge two DataFrames based on latitude, longitude and datetime,
        # rows from df_l1c that do not have a corresponding row in df_l2 are dropped.
        merged_df = pd.merge(self.df_l1c, self.df_l2, on=["Datetime", "Latitude", 'Longitude', "SatelliteZenithAngle"], how='inner')

        # Keep only columns containing variables present in reduced_fields and spectral channels
        reduced_fields = self._get_reduced_fields()
        spectrum_columns = [col for col in merged_df if "Spectrum" in col]
        reduced_df = merged_df.filter(reduced_fields + spectrum_columns)

        # Filter observations to further reduce dataset
        check, filtered_df = self.filter_observations(output_path, reduced_df)

        if check:
            # Save observations
            self._save_merged_products(output_path, filtered_df, delete_obr_files=True)
    
    
    def merge_spectra_and_cloud_products(self):
        """
        Intended to be operated on a specific day, e.g. in the nested year-month-day loop of the default run_pisco.py
        """
        # Load IASI spectra and cloud products
        self.load_data()      
        
        # Correlates measurements, keep matching locations and times of observation
        self.correlate_measurements()
        
        # Merge DataFrames, dropping uncorrelated rows and unwated columns
        self.reduce_fields()