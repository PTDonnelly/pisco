import glob
import os
import pandas as pd
from typing import List, Optional

import numpy as np

class Processor:
    def __init__(self, datapath_out: str, year: str, month: str, day: str, cloud_phase: int):
        self.cloud_phase: int = cloud_phase
        self.datapath_l1c = f"{datapath_out}l1c/{year}/{month}/{day}/"
        self.datapath_l2 = f"{datapath_out}l2/{year}/{month}/{day}/"
        self.datapath_merged = f"{datapath_out}merged/{year}/{month}/{day}/"
        self.df_l1c: object = None
        self.df_l2: object = None
        self.reduced_fields: List[int] = None

    def _get_intermediate_analysis_data_paths(self) -> None:
        """
        Defines the paths to the intermediate analysis data files.
        """
        self.datafile_l1c = f"{self.datapath_l1c}extracted_spectra.csv"
        self.datafile_l2 = f"{self.datapath_l2}cloud_products.csv"

       # Check if L1C and/or L2 data files exist
        if not os.path.exists(self.datafile_l1c) and not os.path.exists(self.datafile_l2):
            raise ValueError('Neither L1C nor L2 data files exist. Nothing to correlate.')
        elif not os.path.exists(self.datafile_l1c):
            raise ValueError('L1C data files do not exist. Cannot correlate.')
        elif not os.path.exists(self.datafile_l2):
            raise ValueError('L2 data files do not exist. Cannot correlate.')
        
    def load_data(self) -> None:
        """
        Opens two DataFrames loaded from the intermediate analysis data files.
        
        """
        # Open csv files
        print("\nLoading L1C spectra and L2 cloud products:")
        self._get_intermediate_analysis_data_paths()
        self.df_l1c, self.df_l2 = pd.read_csv(self.datafile_l1c), pd.read_csv(self.datafile_l2)
        return
    

    def _delete_intermediate_analysis_data(self) -> None:
        """
        Delete the intermediate analysis data files used for correlating spectra and clouds.
        """
        os.remove(self.datafile_l1c)
        os.remove(self.datafile_l2)
        return
    
    def _save_merged_products(self, merged_df: pd.DataFrame) -> None:
        # Create the output directory if it doesn't exist
        os.makedirs(self.datapath_merged, exist_ok=True)

        print(f"Saving spectra to {self.datapath_merged}")
        merged_df.to_csv(f"{self.datapath_merged}spectra_and_cloud_products.csv", index=False, mode='w')

        # # Delete original csv files
        # self._delete_intermediate_analysis_data()
        pass

    def _check_headers(self):
        required_headers = ['Latitude', 'Longitude', 'Datetime', 'Local Time']
        missing_headers_l1c = [header for header in required_headers if header not in self.df_l1c.columns]
        missing_headers_l2 = [header for header in required_headers if header not in self.df_l2.columns]
        if missing_headers_l1c or missing_headers_l2:
            raise ValueError(f"Missing required headers in df_l1c: {missing_headers_l1c} or df_l2: {missing_headers_l2}")
    
    def _get_reduced_fields(self, df_columns: pd.DataFrame) -> None:
        print(type(df_columns))
        print(df_columns)
        exit()
        self.reduced_fields = ["Datetime", "Latitude", 'Longitude', "Satellite Zenith Angle", "Day Night Qualifier", "Cloud Phase 1"]
        return

    def _reduce_fields(self, df: pd.DataFrame) -> None:
        self._get_reduced_fields(df.columns)


    def correlate_measurements(self) -> None:
        """
        Create a single DataFrame for all contemporaneous observations 
        Then separate into day and night observations
        """
        # Check that latitude, longitude, datetime, and local time are present in both file headers 
        self._check_headers()

        # Latitude and longitude values are rounded to 2 decimal places.
        decimal_places = 4
        self.df_l1c[['Latitude', 'Longitude']] = self.df_l1c[['Latitude', 'Longitude']].round(decimal_places)
        self.df_l2[['Latitude', 'Longitude']] = self.df_l2[['Latitude', 'Longitude']].round(decimal_places)
        
        # Merge two DataFrames based on latitude, longitude and datetime,
        # rows from df_l1c that do not have a corresponding row in df_l2 are dropped.
        merged_df = pd.merge(self.df_l1c, self.df_l2, on=['Latitude', 'Longitude', 'Datetime'], how='inner')
        print(merged_df)

        # Drop columns containing variables not present in self.reduced_fields
        reduced_df = self._reduce_fields(merged_df)
        print(reduced_df)
        
        # Save observations
        self._save_merged_products(reduced_df)
        return
    
    def merge_spectra_and_cloud_products(self):
        # Load IASI spectra and cloud products
        self.load_data()      
        
        # Correlates measurements, keep matching locations and times of observation
        self.correlate_measurements()