# Standard library imports
from collections import defaultdict
import logging
import os
from typing import List

# Local application/library specific imports
from pisco import Configurer, Postprocessor

# Configure logging at the module level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gather_daily_statistics(datapath: str, filepaths: List[str], target_variables: List[str], save_files=True):
    """
    Processes and saves daily statistics for specified target variables.

    Parameters:
    - target_variables (list): Variables to process, e.g., ['OLR', 'Phase Fraction'].
    """

    # Initialise a defaultdict of defaultdicts, stores data dynamically based on desired column headers, automatically handles missing keys
    data_dict = defaultdict(lambda: defaultdict(list))
    
    # Create empty list to store date objects
    dates = []

    for filepath in filepaths:
        # Initialise a Postprocessor
        post = Postprocessor(filepath)

        # Prepare DataFrame for analysis
        post.prepare_dataframe()
        
        # Gather results for target variables
        post.process_target_variables(target_variables, data_dict)

        # Append date to list
        date_to_append = post.df['Datetime'].dt.date.iloc[0]
        dates.append(date_to_append)

    if save_files:
        Postprocessor.save_results(data_dict, dates, datapath)

def main():
    """
    """
    # Instantiate a Configurer to get data from config.json
    config = Configurer()

    ### FOR TESTING
    # # The path to the directory that contains the data files
    # config.datapath = "D:\\Data\\iasi\\"
    # datapath = "/data/pdonnelly/iasi/metopb/"

    # # Define temporal range to plot
    # target_years = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
    # target_months = [3, 4, 5]
    # target_days = [day for day in range(1, 32)] # Search all days in each month
    # target_time_range = (target_years, target_months, target_days)
    #####
    
    # Define temporal range to post-process
    target_time_range = Postprocessor.get_target_time_range(config)

    # Find and sort data files
    output_path = os.path.join(config.datapath, config.satellite_identifier)
    files_by_date = Postprocessor.organise_files_by_date(output_path)
    filepaths = Postprocessor.select_files(target_time_range, files_by_date)

    # Define second-order target variables to calculate and plot
    target_variables=['OLR', 'Phase Fraction']

    # Process data files and collect time series for each target variable 
    gather_daily_statistics(output_path, filepaths, target_variables)

if __name__ == "__main__":
    main()