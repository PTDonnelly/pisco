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
        self.df_l1c: object = None
        self.df_l2: object = None

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

    def _save_measurements_by_cloud_phase(self, df_day: pd.DataFrame, df_night: pd.DataFrame, cloud_phase: str) -> None:
        """
        Save the merged DataFrame to a CSV file in the output directory.
        Delete the intermediate l1c and l2 products.
        """
        print(f"Saving {cloud_phase} spectra to {self.datapath_l1c}")
        df_day.to_csv(f"{self.datapath_l1c}extracted_spectra_day_{cloud_phase}.csv", index=False, mode='w')
        df_night.to_csv(f"{self.datapath_l1c}extracted_spectra_night_{cloud_phase}.csv", index=False, mode='w')        

        # # Delete original csv files
        # self._delete_intermediate_analysis_data()
        return
    
    def _get_cloud_phase(self) -> Optional[str]:
        """
        Returns the cloud phase as a string based on the cloud phase value.
        If the retrieved cloud phase is unknown or uncertain, returns None.
        """
        cloud_phase_dictionary = {1: "aqueous", 2: "icy", 3: "mixed", 4: "clear"}
        cloud_phase = cloud_phase_dictionary.get(self.cloud_phase)
        return None if cloud_phase is None else cloud_phase
    
    def _split_measurements_by_cloud_phase(self, df_day: pd.DataFrame, df_night: pd.DataFrame):
        if not self.cloud_phase == "all":
            cloud_phase = self._get_cloud_phase()
            if cloud_phase is None:
                print("Cloud_phase is unknown or uncertain, skipping data.")
            else:
                # Save observations
                self._save_measurements_by_cloud_phase(df_day, df_night, cloud_phase)
        else:
            cloud_phase_dictionary = {1: "aqueous", 2: "icy", 3: "mixed", 4: "clear"}
            for cloud_phase_flag, cloud_phase in cloud_phase_dictionary.items():        
                # Isolate cloud phase
                df_day_phase = df_day[df_day['Cloud Phase 1'] == cloud_phase_flag]
                df_night_phase = df_night[df_night['Cloud Phase 1'] == cloud_phase_flag]

                # Save observations
                self._save_measurements_by_cloud_phase(df_day_phase, df_night_phase, cloud_phase)
        return
    
    def _save_measurements_by_local_time(self, df_day: pd.DataFrame, df_night: pd.DataFrame) -> None:
        print(f"Saving spectra to {self.datapath_l1c}")
        df_day.to_csv(f"{self.datapath_l1c}extracted_spectra_day.csv", index=False, mode='w')
        df_night.to_csv(f"{self.datapath_l1c}extracted_spectra_night.csv", index=False, mode='w')
        pass

    def _split_measurements_by_local_time(self, merged_df: pd.DataFrame) -> None:
        # Split the DataFrame into two based on 'Local Time' column
        merged_df_day = merged_df[merged_df['Local Time'] == True]
        merged_df_night = merged_df[merged_df['Local Time'] == False]
        
        # Drop the 'Local Time' column from both DataFrames
        merged_df_day = merged_df_day.drop(columns=['Local Time'])
        merged_df_night = merged_df_night.drop(columns=['Local Time'])
        
        # Save observations
        self._save_measurements_by_local_time(merged_df_day, merged_df_night)
        
        # Separate into separate datasets for cloud phase
        self._split_measurements_by_cloud_phase(merged_df_day, merged_df_night)
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
        merged_df = pd.merge(self.df_l1c, self.df_l2, on=['Latitude', 'Longitude', 'Datetime', 'Local Time'], how='inner')

        # Convert the DataFrame 'Local Time' column (np.array) to boolean values
        merged_df['Local Time'] = merged_df['Local Time'].astype(bool)

        # Separate into separate datasets for day/night
        self._split_measurements_by_local_time(merged_df)
        return
    
    def correlate_spectra_with_cloud_products(self):
        # Load IASI spectra and cloud products
        self.load_data()      
        
        # Correlates measurements, keep matching locations and times of observation
        self.correlate_measurements()