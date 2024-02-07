# Standard library imports
from collections import defaultdict
from typing import List

# Local application/library specific imports
from pisco import Postprocessor

def gather_daily_statistics(datapath: str, filepaths: List[str], target_variables: List[str]):
    """
    Processes and saves daily statistics for specified target variables.

    Parameters:
    - target_variables (list): Variables to process, e.g., ['OLR', 'Ice Fraction'].
    """

    # Initialise a defaultdict of defaultdicts, stores data dynamically based on desired column headers, automatically handles missing keys
    data_dict = defaultdict(lambda: defaultdict(list))
    
    # Create empty list to store datetime objects
    dates = []

    for filepath in filepaths:
        # Initialise a Postprocessor
        post = Postprocessor(filepath)

        # Prepare DataFrame for analysis
        is_df_prepared = post.prepare_dataframe()

        if is_df_prepared:
            post.process_target_variables(target_variables, data_dict)
        else:
            Postprocessor.append_bad_values(target_variables, data_dict)

        date_to_append = post.df['Datetime'].dt.date.iloc[0] if is_df_prepared else Postprocessor.extract_date_from_filepath(filepath)
        dates.append(date_to_append)

    Postprocessor.save_results(data_dict, dates, datapath)

def main():
    """
    """
    # The path to the directory that contains the data files
    datapath = "D:\\Data\\iasi\\"
    # datapath = "/data/pdonnelly/iasi/metopb_window/"

    # Define temporal range to plot
    target_years = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
    target_months = [3, 4, 5]
    target_days = [day for day in range(1, 32)] # Search all days in each month
    target_range = (target_years, target_months, target_days)
    
    # Define second-order target variables to calculate and plot
    target_variables=['OLR', 'Phase Fraction']

    # Find and sort data files
    files_by_date = Postprocessor.organise_files_by_date(datapath)
    filepaths = Postprocessor.select_files(target_range, files_by_date)

    # Process data files and collect time series for each target variable 
    gather_daily_statistics(datapath, filepaths, target_variables)

if __name__ == "__main__":
    main()