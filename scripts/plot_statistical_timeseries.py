# Standard library imports
import numpy as np
from typing import List

# Third-party library imports
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# Local application/library specific imports
from pisco import Plotter, Postprocessor

def convert_olr_units(df):
    # Identify data columns (excluding 'Date')
    data_columns = df.columns[df.columns != 'Date']

    # Iterate through each data column to apply conversion
    for column in data_columns:
        # Apply conversion to units of mW m^2 only to values not equal to -1
        df[column] = df[column].apply(lambda x: x * 1e6 if x != -1 else x)
    return df

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
        df = convert_olr_units(df)

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
            ylim = [-0.8e-5, -0.4e-5]
        elif var == 'Phase Fraction':
            file_path = f"{plotter.datapath}daily_phase_fraction.csv"
            ylabel = "Phase / Total"
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

            # Drop non-numeric columns before resampling
            numeric_df = df.select_dtypes(include=[np.number])
            # Resample and aggregate only numeric columns
            weekly_mean_df = numeric_df.resample('W').agg(['min', 'mean', 'max'])
            monthly_mean_df = numeric_df.resample('M').agg(['min', 'mean', 'max'])
            # Fix time axis
            weekly_mean_df = make_date_axis_continuous(weekly_mean_df)
            monthly_mean_df = make_date_axis_continuous(monthly_mean_df, number_of_days=15)
            # Drop NaNs (creates gaps between years to avoid unclear datarepresentation)
            weekly_mean_df.dropna(inplace=False)
            monthly_mean_df.dropna(inplace=False)

            # Generate a color palette with enough colors for each column
            palette = sns.color_palette(n_colors=len(df.columns))

            # Track legend entries
            legend_entries = []

            # Plot daily measurements for each data column
            for idx, column in enumerate(df.columns):
                
                # Filter columns to plot
                if not column in ['Date', 'Year', 'Year-Month', 'Year-Month-Day']:  # Skip the 'Date' column
                    if (df[column] == -1).all():
                        continue
                    
                    # Assign a unique color from the palette
                    color = palette[idx]

                    # Scatter plot for daily measurements
                    ax.scatter(df['Year-Month-Day'], df[column], label=f'Daily {column}', marker='.', s=1, color=color, alpha=0.75)

                    # Line plot for weekly mean
                    weekly_color = np.clip(np.array(color) * 0.9, 0, 1)  # Darken the color
                    weekly_line = ax.plot(weekly_mean_df['Year-Month-Day'], weekly_mean_df[column]['mean'], label=f'Weekly Mean {column}', ls='-', lw=1, color=weekly_color)
                    
                    # Line plot for monthly mean
                    monthly_color = np.clip(np.array(color) * 0.8, 0, 1)  # Darken the color further
                    monthly_line = ax.plot(monthly_mean_df['Year-Month-Day'], monthly_mean_df[column]['mean'], label=f'Monthly Mean {column}', ls='-', lw=2, marker='o', markersize=4, color=monthly_color)
                    
                    # Add legend entry for this column
                    legend_entries.append((weekly_line, monthly_line))

            # Create custom legend handles
            weekly_handle = mlines.Line2D([], [], color='black', linestyle='--', label='Weekly Mean')
            monthly_handle = mlines.Line2D([], [], color='black', linestyle='-', marker='o', label='Monthly Mean')

            # Add column color legend entries
            column_handles = []

            for entry in legend_entries:
                print(entry)
                # entry is a tuple containing line objects for a specific column
                for obj in entry:  # Iterate over each object within the tuple
                    if hasattr(obj, 'get_color') and hasattr(obj, 'get_linestyle') and hasattr(obj, 'get_label'):
                        # Extract the properties from the line object to create a custom handle
                        color = obj.get_color()
                        linestyle = obj.get_linestyle()
                        # The label might need to be adjusted depending on how you want it to appear
                        label = obj.get_label().split()[0]  # Adjust this as needed

                        # Create a custom legend handle for this line object
                        handle = mlines.Line2D([], [], color=color, linestyle=linestyle, label=label)
                        column_handles.append(handle)
                        
                        # Combine all legend handles
                        all_handles = column_handles + [weekly_handle, monthly_handle]

                        # Add the legend to the plot
                        ax.legend(handles=all_handles, bbox_to_anchor=(1.05, 1), loc='upper left')




            ax.set_xticks(df['Year'].unique())

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

