import glob
import os
import pandas as pd
from typing import Optional

class L1C_L2_Correlator:
    def __init__(self, datapath_out: str, year: str, month: str, day: str, cloud_phase: int):
        self.cloud_phase: int = cloud_phase
        self.datapath_l1c = f"{datapath_out}l1c/{year}/{month}/{day}/"
        self.datapath_l2 = f"{datapath_out}l2/{year}/{month}/{day}/"
        self.df_l1c: object = None
        self.df_l2: object = None
        self.merged_df: object = None
        self.merged_datafile: str = None

    def gather_files(self) -> None:
        """
        Gather all CSV files from a specified directory, combine them into a single dataframe, 
        and save this combined dataframe as a new CSV file in the same directory.

        This function scans through all files in the directory specified by `datapath_out`,
        reads each CSV file into a dataframe, concatenates these dataframes into a single dataframe,
        and saves this combined dataframe as a new CSV file ('cloud_products.csv') in the same directory.

        Args:
        ----------
        datapath_out : str
            The path of the directory that contains the CSV files to gather.

        Notes
        -----
        The CSV files in `datapath_out` must all have the same columns for the concatenation to work.
        The combined dataframe will have a new index, not preserving the original indices from separate dataframes.
        The row indices are not included in the saved CSV file.
        """
        print("Combining L2 cloud products into single file")
        # Create an empty list to hold dataframes
        df_list = []
        for datafile_out in os.scandir(self.datapath_l2):
            # Check that entry is a file
            if datafile_out.is_file():
                # Open csv as data frame and append
                df = pd.read_csv(datafile_out.path)
                df_list.append(df)

        # Concatenate all dataframes in the list
        combined_df = pd.concat(df_list, ignore_index=True)

        # Delete all original csv files in data_path_out
        [os.remove(file) for file in glob.glob(os.path.join(self.datapath_l2, '*.csv'))]

        # Save as single csv
        combined_df.to_csv(os.path.join(self.datapath_l2, 'cloud_products.csv'), index=False)
        return


    def _get_intermediate_analysis_data_paths(self) -> None:
        """
        Defines the paths to the intermediate analysis data files.
        """
        self.datafile_l1c = f"{self.datapath_l1c}extracted_spectra.csv"
        self.datafile_l2 = f"{self.datapath_l2}cloud_products.csv"

       # Check if L1C and/or L2 data files exist
        if not os.path.exists(self.datafile_l1c) and not os.path.exists(self.datafile_l2):
            raise ValueError('Neither L1C nor L2 data files exist. Nothing to correlate.')
    
    def load_data(self) -> None:
        """
        Opens two DataFrames loaded from the intermediate analysis data files.
        
        """
        # Open csv files
        print("Loading L1C spectra and L2 cloud products:")
        self._get_intermediate_analysis_data_paths()
        self.df_l1c, self.df_l2 = pd.read_csv(self.datafile_l1c), pd.read_csv(self.datafile_l2)
        return
    

    def _check_headers(self):
        required_headers = ['Latitude', 'Longitude', 'Datetime']#, 'Local Time']
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
        self.merged_df = pd.merge(self.df_l1c, self.df_l2, on=['Latitude', 'Longitude', 'Datetime'], how='inner')

        # # Convert the DataFrame 'Local Time' column (np.array) to boolean values
        # merged_df['Local Time'] = merged_df['Local Time'].astype(bool)
        # # Split the DataFrame into two based on 'Local Time' column
        # merged_df_day = merged_df[merged_df['Local Time'] == True]
        # merged_df_night = merged_df[merged_df['Local Time'] == False]
        # # # Drop the 'Local Time' column from both DataFrames
        # merged_df_day = merged_df_day.drop(columns=['Local Time'])
        # merged_df_night = merged_df_night.drop(columns=['Local Time'])
        # # Remove 'Local Time' from the header list
        # header.remove('Local Time')
        return #merged_df_day.dropna(), merged_df_night.dropna()
    

    def _delete_intermediate_analysis_data(self) -> None:
        """
        Delete the intermediate analysis data files used for correlating spectra and clouds.
        """
        os.remove(self.datafile_l1c)
        os.remove(self.datafile_l2)

    def _get_cloud_phase(self) -> Optional[str]:
        """
        Returns the cloud phase as a string based on the cloud phase value.
        If the retrieved cloud phase is unknown or uncertain, returns None.
        """
        cloud_phase_dictionary = {1: "aqueous", 2: "icy", 3: "mixed", 4: "clear"}
        cloud_phase = cloud_phase_dictionary.get(self.cloud_phase)
        return None if cloud_phase is None else cloud_phase

    def save_merged_data(self) -> None:
        """
        Save the merged DataFrame to a CSV file in the output directory.
        If the output directory is unknown (because the cloud phase is unknown), print a message and return.
        Delete the intermediate l1c and l2 products.
        """
        cloud_phase = self._get_cloud_phase()
        if cloud_phase is None:
            print("Cloud_phase is unknown or uncertain, skipping data.")
        else:
            print(f"Saving {cloud_phase} spectra to {self.datapath_l1c}")
            self.merged_datafile = f"{self.datapath_l1c}extracted_spectra_{cloud_phase}.csv"
            self.merged_df.to_csv(self.merged_datafile, index=False, mode='w')
            
            # # # Save the DataFrame to a file in csv format, split by local time
            # # df.to_hdf(f"{datapath_out}{datafile_out}.h5", key='df', mode='w')
            # merged_df_day.to_csv(f"{datapath_out}day_extracted_spectra.csv", index=False, mode='w')
            # merged_df_night.to_csv(f"{datapath_out}night_extracted_spectra.csv", index=False, mode='w')
        
        # Delete original csv files
        # self._delete_intermediate_analysis_data()
        return
    

    def preview_merged_data(self) -> None:
        # Load the CSV file
        df = pd.read_csv(self.merged_datafile)

        # Get the number of rows (records) and columns in the DataFrame
        num_rows, num_cols = df.shape

        print(f"The dataframe has {num_rows} rows and {num_cols} columns.")
        return