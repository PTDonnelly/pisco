# Standard library imports
from typing import List

# Third-party library imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Local application/library specific imports
from pisco import Plotter, Postprocessor

def load_data(file_path, var):
    """
    Loads and sorts data from a .npy file.

    Parameters:
    - file_path (str): Path to the .npy file.
    - column_name (str): Name of the column for the data values (default is 'Value').

    Returns:
    - df (pd.DataFrame): DataFrame containing Date and data entries
    """
    # Read .csv as DataFrame
    df = pd.read_csv(file_path, sep=',')

    # Ensure 'Date' is set as the DataFrame index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)


    if var == 'OLR':
        # Convert outgoing longwave radiation to units of mW m^2
        df.loc[:, df.columns != 'Date'] = df.loc[:, df.columns != 'Date'].where(df.loc[:, df.columns != 'Date'] == -1, df.loc[:, df.columns != 'Date'] * 1e6)

    return df

def add_grey_box(ax, df, plot_type):
    """
    Adds grey boxes to the plot for every other year.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to add the grey boxes to.
    - df (pd.DataFrame): DataFrame with 'Year' column.
    """
    unique_years = sorted(df['Year'].unique())
    for i, year in enumerate(unique_years):
        
        if plot_type == 'violin':
            if i % 2 == 0:
                ax.axvspan(i-0.5, i+0.5, color='grey', alpha=0.2, zorder=0)
        elif plot_type == 'line':
            if i % 2 == 0:
                ax.axvspan(year, year+1, color='grey', alpha=0.2, zorder=0)

def make_date_axis_continuous(df, number_of_months=3, number_of_days=0):
    """For converting a Date index in DataFrame to a continuous numerical axis for plotting."""
    df['Year-Month-Day'] = df.index.year + ((df.index.month - number_of_months) / number_of_months) + ((df.index.day - number_of_days)/ 100)
    return df

def plot_statistical_timeseries(plotter: object, target_variables: List[str], plot_type: str):
    """
    Loads data from a .csv file, filters for spring months (March, April, May),
    and generates a violin plot with strip plot overlay for each year.

    Parameters:
    - plotter (object): An instance with methods for data handling and plotting configurations.
    - file_path (str): Path to the .csv file containing the data.
    - column_name (str): Name of the column for the data values, defaulting to 'Value'.
    - plot_title (str): Title for the plot.
    """
    # Load, sort, and return the sorted DataFrame for each target variable
    for var in target_variables:
        if var == 'OLR':
            file_path = f"{plotter.datapath}daily_olr.csv"
            ylabel = fr"{var} mW m$^{-2}$"
            ylim = [-1.2e-2, 0]
        elif var == 'Ice Fraction':
            file_path = f"{plotter.datapath}daily_ice_fraction.csv"
            ylabel = "Ice / Total"
            ylim = [0, 1]

        # Load csv data
        df = load_data(file_path, var)

        # Create a subplot layout
        _, ax = plt.subplots(figsize=(8, 4))
        
        if plot_type == 'violin':
            # Create formatting inputs
            xlabel = 'Year'

            # Add 'Year' and 'Month'
            df['Year'] = df.index.year
            df['Month'] = df.index.month_name().str[:3]

            # Violin Plot with Colors: visualises the distribution of data values for each spring month across years
            sns.violinplot(x='Year', y=var, hue='Month', data=df, ax=ax, palette="muted", split=False)

            # Strip Plot: adds individual data points to the violin plot for detailed data visualization
            sns.stripplot(x='Year', y=var, hue='Month', data=df, ax=ax, palette='dark:k', size=3, jitter=False, dodge=True)

            # Add grey box for visual separation of every other year for enhanced readability
            add_grey_box(ax, df, plot_type)

            # Handling the legend to ensure clarity in distinguishing between different months
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:3], labels[:3], title='Month')
        elif plot_type == 'line':
            # Create formatting inputs
            xlabel = 'Date'
            
            df = make_date_axis_continuous(df)
            df['Year'] = df.index.year
            df['Year-Month'] = df.index.strftime('%Y-%m')

            # Calculate temporal averages
            weekly_mean_df = df.resample('W').agg(['min', 'mean', 'max'])
            monthly_mean_df = df.resample('M').agg(['min', 'mean', 'max'])
            weekly_mean_df = make_date_axis_continuous(weekly_mean_df)
            monthly_mean_df = make_date_axis_continuous(monthly_mean_df, number_of_days=15)
            weekly_mean_df.dropna(inplace=False)
            monthly_mean_df.dropna(inplace=False)
            
            print(weekly_mean_df.head())

            exit()

            # Plot daily measurements
            ax.scatter(df['Year-Month-Day'], df, label='Daily', s=1, color='grey')
            # Plot temporal averages
            ax.plot(weekly_mean_df['Year-Month-Day'], weekly_mean_df['mean'], label='Weekly Mean', ls='-', lw=1, color='black')
            ax.plot(monthly_mean_df['Year-Month-Day'], monthly_mean_df['mean'], label='Monthly Mean', ls='-', lw=2, marker='o', markersize=4, color='red')
            
            ax.set_xticks(df['Year'].unique())
            ax.legend()

            # Add grey box for visual separation of every other year for enhanced readability
            add_grey_box(ax, df, plot_type)

        # Customizing the plot with titles and labels
        ax.set_title(f"MAM Average {var} at Nadir")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.grid(axis='y', linestyle=':', color='k')
        ax.tick_params(axis='both', labelsize=plotter.fontsize)

        # Save the plot to a file and close the plotting context to free up memory
        plt.tight_layout()
        plt.savefig(f"{plotter.datapath}daily_{var.lower().replace(' ', '_')}_by_phase_{plot_type}.png", dpi=300)
        plt.close()


def main():
    """
    """
    # The path to the directory that contains the data files
    datapath = "D:\\Data\\iasi\\"
    # datapath = "/data/pdonnelly/iasi/metopb_window/"

    # Define plotting parameters
    fontsize = 10
    dpi = 300
    
    # Instantiate the Plotter and organise files
    plotter = Plotter(datapath, fontsize, dpi)
    
    # Define second-order target variables to calculate and plot
    target_variables=['OLR', 'Phase Fraction']

    # Plot data
    plot_statistical_timeseries(plotter, target_variables, plot_type='line')

if __name__ == "__main__":
    main()

