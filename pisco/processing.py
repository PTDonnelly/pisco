import os
import pandas as pd
from typing import List

class Processor:
    def __init__(self, ex: Extractor):
        self.cloud_phase: int = ex.config.cloud_phase
        self.output_format: str = ex.config.output_format
        self.datapath_l1c = f"{ex.datapath_out}l1c/{ex.year}/{ex.month}/{ex.day}/"
        self.datapath_l2 = f"{ex.datapath_out}l2/{ex.year}/{ex.month}/{ex.day}/"
        self.datapath_merged = f"{ex.datapath_out}merged/{ex.year}/{ex.month}/{ex.day}/"
        self.df_l1c: object = None
        self.df_l2: object = None

    def _get_intermediate_analysis_data_paths(self) -> None:
        """
        Defines the paths to the intermediate analysis data files.
        """
        self.datafile_l1c = f"{self.datapath_l1c}extracted_spectra.{self.output_format}"
        self.datafile_l2 = f"{self.datapath_l2}cloud_products.{self.output_format}"
    
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
        self.df_l1c = pd.read_csv(self.datafile_l1c, sep="\t")
        self.df_l2 = pd.read_csv(self.datafile_l2, sep="\t")
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

        # Latitude and longitude values are rounded to 2 decimal places.
        decimal_places = 4
        self.df_l1c[['Latitude', 'Longitude']] = self.df_l1c[['Latitude', 'Longitude']].round(decimal_places)
        self.df_l2[['Latitude', 'Longitude']] = self.df_l2[['Latitude', 'Longitude']].round(decimal_places)
        return
    

    def _delete_intermediate_analysis_data(self) -> None:
        """
        Delete the intermediate analysis data files used for correlating spectra and clouds.
        """
        os.remove(self.datafile_l1c)
        os.remove(self.datafile_l2)
        return
    
    def _save_merged_products(self, merged_df: pd.DataFrame, delete_obr_files: bool = False) -> None:
        # Create the output directory if it doesn't exist
        os.makedirs(self.datapath_merged, exist_ok=True)

        print(f"Saving spectra to {self.datapath_merged}")
        merged_df.to_csv(f"{self.datapath_merged}spectra_and_cloud_products.csv", index=False, mode='w')

        if delete_obr_files == True:
            # Delete original csv files
            self._delete_intermediate_analysis_data()
    
    @staticmethod
    def _get_reduced_fields() -> List[str]:
        reduced_fields = ["Datetime", "Latitude", 'Longitude', "Satellite Zenith Angle", "Day Night Qualifier", "Cloud Phase 1"]
        return reduced_fields
    
    def reduce_fields(self) -> None:
        # Merge two DataFrames based on latitude, longitude and datetime,
        # rows from df_l1c that do not have a corresponding row in df_l2 are dropped.
        merged_df = pd.merge(self.df_l1c, self.df_l2, on=['Latitude', 'Longitude', 'Datetime'], how='inner')
        
        # Keep only columns containing variables present in reduced_fields and spectral channels
        reduced_fields = self._get_reduced_fields()
        spectrum_columns = [col for col in merged_df if "Spectrum " in col]
        reduced_df = merged_df.filter(reduced_fields + spectrum_columns)
        
        # Save observations
        self._save_merged_products(reduced_df, delete_obr_files=False)
    
    
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