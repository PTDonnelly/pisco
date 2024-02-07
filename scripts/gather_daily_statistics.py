# Standard library imports
from typing import List

import line_profiler as LineProfiler

# Local application/library specific imports
from pisco import Plotter, Processor, Postprocessor

def gather_daily_statistics(datapath: str, filepaths: List[str], target_variables: List[str]):
    """
    Processes and saves daily statistics for specified target variables.

    Parameters:
    - target_variables (list): Variables to process, e.g., ['OLR', 'Ice Fraction'].
    """

    # Initialise a dictionary to store the data for each target variable
    data_dict = {var: [] for var in target_variables}
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
    # datapath = "D:\\Data\\iasi\\"
    datapath = "/data/pdonnelly/iasi/metopb_window/"

    # Define temporal range to plot
    target_years = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
    target_months = [3, 4, 5]
    target_days = [day for day in range(1, 32)] # Search all days in each month
    target_range = (target_years, target_months, target_days)
    
    # Define second-order target variables to calculate and plot
    target_variables=['OLR', 'Ice Fraction']

    # Find and sort data files
    files_by_date = Postprocessor.organise_files_by_date(datapath)
    filepaths = Postprocessor.select_files(target_range, files_by_date)

    # Plot data
    gather_daily_statistics(datapath, filepaths, target_variables)

if __name__ == "__main__":
    # lp = LineProfiler()
    # lp_wrapper = lp(main())
    # lp_wrapper()
    # lp.print_stats()

    lp = LineProfiler()
    lp.add_function(main)  # Add the function you want to profile
    lp.run('main()')  # Execute the function within the LineProfiler context
    lp.print_stats()  # Print the profiling results