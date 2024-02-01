# Standard library imports
from typing import List

# Third-party library imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap

# Local application/library specific imports
from pisco import Plotter, Spectrum, Geographic


def plot_spatial_distribution_scatter(datapath: str):
    # Instantiate the Plotter and organise files
    plotter = Plotter(datapath)
    plotter.organise_files_by_date()

    # Define temporal range to plot
    target_year = '2019'
    target_month = '01'
    target_days = [str(day).zfill(2) for day in range(1, 32)]

    # Define spatial range to plot
    lat_range = (30, 60)
    lon_range = (-60, 0)

    # Select files in time range
    all_files = plotter.select_files(target_year, target_month, target_days)
    day_icy_files = plotter.select_files(target_year, target_month, target_days, "day_icy")

    # Define plotting parameters
    fontsize = 7
    dpi = 540

    png_files = []
    # Walk through the directory
    for i, (all_file, day_icy_file) in enumerate(zip(all_files, day_icy_files)):
        # Load the data from the file into a pandas DataFrame
        all_data = pd.read_csv(all_file, usecols=['Longitude', 'Latitude'])
        day_icy_data = pd.read_csv(day_icy_file, usecols=['Longitude', 'Latitude'])

        # Initialise a new figure for the plot
        plt.figure(figsize=(7, 7), dpi=dpi)
        # Create a basemap of the world
        m = Basemap(projection='cyl', resolution="l", llcrnrlon=lon_range[0], llcrnrlat=lat_range[0], urcrnrlon=lon_range[1]+0.1, urcrnrlat=lat_range[1])
        # Draw coastlines and country borders
        m.drawcoastlines()
        # Add spatial grid
        meridians = np.arange(lon_range[0], lon_range[1]+1, 10)
        parallels = np.arange(lat_range[0], lat_range[1]+1, 10)
        m.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.5, dashes=[1, 1], fontsize=fontsize)
        m.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.5, dashes=[1, 1], fontsize=fontsize)

        # Plot the observations on the map
        x, y = m(all_data['Longitude'].values, all_data['Latitude'].values)
        m.scatter(x, y, latlon=True, marker=".", s=1, color="silver", alpha=0.8)
        # Plot the observations on the map
        x, y = m(day_icy_data['Longitude'].values, day_icy_data['Latitude'].values)
        m.scatter(x, y, latlon=True, marker=".", s=1.25, color="dodgerblue")

        # Add a title to the plot
        plt.title(f"Icy Spectra in the North Atlantic: {target_year}-{target_month}-{target_days[i]}", fontsize=fontsize+1)
        # Add labels to the axes
        plt.xlabel('Longitude (deg)', fontsize=fontsize+1, labelpad=20)
        plt.ylabel('Latitude (deg)', fontsize=fontsize+1, labelpad=20)
        plt.xticks()
        plt.yticks()

        # Save figure
        png_file = f"{datapath}/spatial_distribution_{i}.png"
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
        plt.close()

        # Append filename to list of png files
        png_files.append(png_file)

    # Convert all individual pngs to animated gif
    plotter.png_to_gif(f"{datapath}/spatial_distribution.gif", png_files)

def plot_spatial_distribution_2Dhist(plotter: object):        
    # Use Plotter to organise files
    plotter.organise_files_by_date()

    # Select files in time range
    datafiles = plotter.select_files()

    # Create an instance of the Geographic class
    geographic = Geographic()
    
    # Define plotting parameters
    fontsize = 8
    dpi = 150
    png_files = []

    # Define spatial range to plot
    lat_range = (30, 60)
    lon_range = (-60, 0)
    
    # Define file groups and the plot attributes
    plot_params = {
        "day_icy": {"local_time": "day", "phase": "icy", "cmap": "Blues", "title": "Ice Phase: Day"},
        "night_icy": {"local_time": "night", "phase": "icy", "cmap": "Blues", "title": "Ice Phase: Night"},
        "day_liquid": {"local_time": "day", "phase": "liquid", "cmap": "Greens", "title": "Liquid Phase: Day"},
        "night_liquid": {"local_time": "night", "phase": "liquid", "cmap": "Greens", "title": "Liquid Phase: Night"},
        "day_mixed": {"local_time": "day", "phase": "mixed", "cmap": "Purples", "title": "Mixed: Day"},
        "night_mixed": {"local_time": "night", "phase": "mixed", "cmap": "Purples", "title": "Mixed: Night"},
    }

    for ifile, datafile in enumerate(datafiles):
        # Initialise a new figure for the plot with three subplots
        fig = plt.figure(figsize=(10, 8), dpi=dpi)
        axes = gridspec.GridSpec(3, 2, figure=fig).subplots()
        fig.suptitle(f"IASI Spectra in the North Atlantic: {plotter.target_year}-{plotter.target_month}-{plotter.target_days[ifile]}", fontsize=fontsize+5, y=0.95)
        
        # Get current file and load data
        df = pd.read_csv(datafile, usecols=['Longitude', 'Latitude', 'CloudPhase1', 'Day Night Qualifier'])

        for iax, (ax, (group, attrs)) in enumerate(zip(axes.flat, plot_params.items())):
            local_time = attrs["local_time"]
            phase = attrs["phase"]
            sub_df = plotter.extract_by_cloud_phase_and_day_night(df, {local_time: [phase]})
            
            # Create a basemap using the Geographic method
            basemap = geographic.create_basemap(ax, lon_range, lat_range, fontsize)

            # Check if DataFrame contains information
            if not plotter.check_df(datafile, sub_df, phase):
                # Plot the observations on the map as a 2D histogram
                plotter.plot_geographical_contour(sub_df, lon_range, lat_range, basemap, attrs["cmap"])

                # Add a title to the plot
                ax.set_title(attrs["title"], fontsize=fontsize+1)

        # Save figure and store png filename for gif conversion
        filename = "spatial_distribution"
        png_files = plotter.finalise_plot(filename, ifile, png_files, dpi, hspace=0.35, wspace=0.1)

    # Convert all individual pngs to animated gif
    plotter.png_to_gif(f"{plotter.datapath}/{filename}.gif", png_files)

def plot_spatial_distribution_unity(datapath: str):    
    def group_data_by_cloud_phase(files: List[str], ifile: int) -> dict:
        """
        Returns a dictionary where each key corresponds to a group from 'file_groups',
        and each value is a DataFrame grouped by truncated longitude, latitude and cloud phase.

        Args:
        file_groups: A dictionary where each key is a group name and the value is another 
        dictionary with keys 'files', which contains a list of file paths.

        Returns:
        A dictionary where each key corresponds to a group from 'file_groups', and each
        value is a DataFrame grouped by truncated longitude, latitude and cloud phase.
        """
        df_all_grouped = []
        for file in files:
            # Get current file and load data
            df = pd.read_csv(file[ifile], usecols=['Longitude', 'Latitude', 'CloudPhase1'])

            # Truncate latitude and longitude values onto a 10-degree grid
            df['Longitude_truncated'] = (df['Longitude'] // 10 * 10).astype(int)  # Group longitudes into 10 degree bins
            df['Latitude_truncated'] = (df['Latitude'] // 10 * 10).astype(int)  # Group latitudes into 10 degree bins

            df_grouped = df.groupby(['Longitude_truncated', 'Latitude_truncated', 'CloudPhase1']).size().reset_index(name='Counts')

            # Append the grouped dataframe to the overall dataframe
            df_all_grouped.append(df_grouped)
        return pd.concat(df_all_grouped, ignore_index=True)
    
    def plot_grouped_data(ax: object, df_all_grouped: pd.DataFrame, cloud_phase_colors: List[str]) -> None:
        # Iterate over unique combinations of Latitude and Longitude
        for (lat, lon), df_sub in df_all_grouped.groupby(['Latitude_truncated', 'Longitude_truncated']):

            # Compute normalised coordinates for the subplot within the main figure (note: this depends on your actual lat-lon ranges)
            # Assuming the longitude ranges from -180 to 180 and the latitude from -90 to 90
            normalised_lon = (lon + 60) / 60
            normalised_lat = (lat - 30) / 30
            
            # Create an inset_axes object that corresponds to a subplot within the grid cell
            ax_sub = ax.inset_axes([normalized_lon, normalized_lat, 1/6, 1/3])
            
            # Create a bar plot for each cloud phase within the group
            for phase in [1, 2, 3]:
                # Get the count for the current cloud phase
                count = df_sub.loc[df_sub['CloudPhase1'] == phase, 'Counts'] / df_sub['Counts'].sum()
                count = count.values[0] if not count.empty else 0  # default count to 0 if phase not present
                
                # Create the bar plot
                ax_sub.bar(phase, count, color=cloud_phase_colors.get(phase))      
                
                # Set the background color to be transparent
                ax_sub.set_facecolor('none')

                # Remove axis labels
                ax_sub.set_xticks([])
                ax_sub.set_yticks([])

                # Remove the box around the subplot
                for spine in ax_sub.spines.values():
                    spine.set_visible(False)
        return
    
    def plot_scatter(m: Basemap, all_files: dict, day_icy_files: dict, night_icy_files, ifile):
        # Get current file and load data
        all_data = pd.read_csv(all_files[ifile], usecols=['Longitude', 'Latitude'])
        day_icy_data = pd.read_csv(day_icy_files[ifile], usecols=['Longitude', 'Latitude'])
        night_icy_data = pd.read_csv(night_icy_files[ifile], usecols=['Longitude', 'Latitude'])

        # Plot the observations on the map
        x, y = m(all_data['Longitude'].values, all_data['Latitude'].values)
        m.scatter(x, y, latlon=True, marker=".", s=1, color="silver", alpha=0.8)
       
        # Plot the observations on the map
        x, y = m(day_icy_data['Longitude'].values, day_icy_data['Latitude'].values)
        m.scatter(x, y, latlon=True, marker=".", s=1.25, color="crimson")
        
        # Plot the observations on the map
        x, y = m(night_icy_data['Longitude'].values, night_icy_data['Latitude'].values)
        m.scatter(x, y, latlon=True, marker=".", s=1.25, color="crimson")

    # Instantiate the Plotter and organise files
    plotter = Plotter(datapath)
    plotter.organize_files_by_date()

    # Define temporal range to plot
    target_year = '2019'
    target_month = '01'
    target_days = [str(day).zfill(2) for day in range(1, 12)]

    # Define spatial range to plot
    lat_range = (30, 60)
    lon_range = (-60, 0)

    # Define plotting parameters
    fontsize = 7
    dpi = 540
    png_files = []

    # Select files in time range
    all_files = plotter.select_files(target_year, target_month, target_days)
    day_icy_files = plotter.select_files(target_year, target_month, target_days, target_file_part='day_icy')
    night_icy_files = plotter.select_files(target_year, target_month, target_days, target_file_part='night_icy')
    day_liquid_files = plotter.select_files(target_year, target_month, target_days, target_file_part='day_aqueous')
    night_liquid_files = plotter.select_files(target_year, target_month, target_days, target_file_part='night_aqueous')
    day_mixed_files = plotter.select_files(target_year, target_month, target_days, target_file_part='day_mixed')
    night_mixed_files = plotter.select_files(target_year, target_month, target_days, target_file_part='night_mixed')

    cloud_phase_colors = {2: 'royalblue', 1: 'forestgreen', 3: 'darkorchid'}

    for ifile in range(len(target_days)):
        # Get grouped data
        files = [day_icy_files, night_icy_files, day_liquid_files, night_liquid_files, day_mixed_files, night_mixed_files]
        df_all_grouped = group_data_by_cloud_phase(files, ifile)

        # Initialize a new figure for the plot with three subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), dpi=dpi)

        for iax, ax in enumerate(axs.flat):
            if iax == 0:
                # Create a basemap of the world
                m = plotter.create_basemap(lon_range, lat_range, ax, fontsize)
                plot_scatter(m, all_files, day_icy_files, night_icy_files, ifile)
            elif iax == 1:
                # Create a basemap of the world
                m = plotter.create_basemap(lon_range, lat_range, ax, fontsize)
                plot_grouped_data(ax, df_all_grouped, cloud_phase_colors)
            elif iax == 2:
                pass
        
        # Final adjustments
        plt.subplots_adjust(hspace=0.2, wspace=0.1)

        # Save figure
        png_file = f"{datapath}/unity_{ifile}.png"
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
        plt.close()

        # Append filename to list of png files
        png_files.append(png_file)

    # Convert all individual pngs to animated gif
    plotter.png_to_gif(f"{datapath}/unity.gif", png_files)

def plot_spectral_distributon(plotter: object):
    # Use Plotter to organise files
    plotter.organize_files_by_date()

    # Select files in time range
    datafiles = plotter.select_files()
    
    png_files = []
    titles = ['Spectrum', 'Normalised Residuals', 'Normalised Residuals']
    phases = ['icy', 'liquid', 'mixed']
    cmaps = ['Blues', 'Greens', 'Purples']

    for ifile, datafile in enumerate(datafiles):
        # Initialize a new figure for the plot with three subplots
        fig = plt.figure(figsize=(15, 9), dpi=plotter.dpi)
        gs = gridspec.GridSpec(3, 5, figure=fig)
        fig.suptitle(f"IASI Spectra in the North Atlantic: {plotter.target_year}-{plotter.target_month}-{plotter.target_days[ifile]}", fontsize=plotter.fontsize+5, y=0.95)

        # Get current file and load data
        print(f"\nLoading datafile: {datafile}")
        df = pd.read_csv(datafile)

        for irow, (phase, cmap) in enumerate(zip(phases, cmaps)):
            # Create smaller dataframe of spectra by cloud phase and local time, and convert to mW
            print(f"Scanning {phase} spectra")
            sub_df = plotter.extract_by_cloud_phase_and_day_night(df, {'day': [phase], 'night': [phase]}).filter(regex='Spectrum ') * 1000
            
            # Set plotting color for 1-D plots
            color = plotter.get_color_from_cmap(cmap)

            # Check if DataFrame contains information
            if not plotter.check_df(datafile, sub_df, phase):
                # Calculate plotting data
                wavenumbers = plotter.get_dataframe_spectral_grid(sub_df)

                # Create plotting objects
                spectrum = Spectrum(sub_df, wavenumbers)
                spectrum.build()

                for icol in range(5):
                    if icol == 0:
                        ax = fig.add_subplot(gs[irow, icol:icol+2])
                        ax = spectrum.plot_spectrum(ax, phase=phase, color=color)
                        xlabel = r'Wavenumber (cm$^{-1}$)'
                        ylabel = r'Radiance ($mWm^{-2}srm^{-1}m$)'
                    elif icol == 2:
                        ax = fig.add_subplot(gs[irow, icol:icol+2])
                        spectrum.compute_mean_histogram_2d()
                        ax = spectrum.plot_histogram_2d(ax, cmap=cmap)
                        xlabel = r'Wavenumber (cm$^{-1}$)'
                        ylabel = r'Normalised Residual'
                    elif icol == 4:
                        ax = fig.add_subplot(gs[irow, icol])
                        ax = spectrum.plot_residuals_histogram_1d(ax, color=color)
                        xlabel = r'Normalised Residual'
                        ylabel = r'Probability Density'

                    ax.grid(axis='both', color='k', lw=0.5, ls=':')
                    if irow == 0:
                        if icol in [0, 1]:  # For the first two columns
                            ax.set_title(titles[0], fontsize=plotter.fontsize+1)
                        elif icol in [2, 3]:  # For the next two columns
                            ax.set_title(titles[1], fontsize=plotter.fontsize+1)
                        elif icol == 4:  # For the last column
                            ax.set_title(titles[2], fontsize=plotter.fontsize+1)
                    if irow == 2:
                        ax.set_xlabel(xlabel, labelpad=1, fontsize=plotter.fontsize)
                    ax.set_ylabel(ylabel, labelpad=2, fontsize=plotter.fontsize)

        # Save figure and store png filename for gif conversion
        filename = "spectral_distribution_hist"
        png_files = plotter.finalise_plot(filename, ifile, png_files, plotter.dpi, hspace=0.35, wspace=0.4)

    # Convert all individual pngs to animated gif
    plotter.png_to_gif(f"{plotter.datapath}/{filename}.gif", png_files)


def prepare_dataframe(datafile, df, maximum_zenith_angle=5):
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
        # Proceed with DataFrame manipulations if all required columns are present
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y%m%d%H%M')
        df = df[df['CloudPhase1'] != -1]
        df = df[df['SatelliteZenithAngle'] < maximum_zenith_angle]
        return True, df
    else:
        print(f"\Skipping DataFrame: {datafile}")
        return False, None

def get_outgoing_longwave_radiation(plotter, df):
    # Retrieve IASI spectral grid and radiance form the DataFrame
    wavenumbers = plotter.get_dataframe_spectral_grid(df)
    radiance = df[[col for col in df.columns if 'Spectrum' in col]].values
    
    # Convert wavenumbers to wavelengths in meters
    wavelengths = 1e-2 / np.array(wavenumbers)  # Conversion from cm^-1 to m
    # Convert radiance to SI units: W/m^2/sr/m
    radiance_si = radiance * 1e-3  # Convert from mW to W
    
    # Integrate the radiance over the wavelength and sum integral elements
    olr_integrals = np.trapz(radiance_si, wavelengths, axis=1)
    olr_total = np.sum(olr_integrals)
    return olr_total

def get_ice_fraction(df):
    # Re-format DataFrame to get the individual counts of each Cloud Phase per Datetime
    pivot_df = df.groupby([df['Datetime'].dt.date, 'CloudPhase1']).size().unstack(fill_value=0)
    
    # Calculate total number of measurements for the entire day
    total_measurements = pivot_df.sum(axis=1)
    # Calculate proportion of measurements for the entire day flagged as "icy"
    ice_count = pivot_df.get(2, 0).sum()
    return ice_count / total_measurements

def gather_daily_statistics(plotter: object, target_variables: List[str]):
    """
    Processes data from a series of data files for specified target variables, such as OLR or Ice Fraction,
    and saves the results in separate .npy files named after each target variable.

    Parameters:
    plotter (object): An instance of the Plotter class with methods for data handling.
    target_variables (list): List of target variables to process, e.g., ['OLR', 'Ice Fraction'].
    """
    plotter.organise_files_by_date()
    datafiles = plotter.select_files()

    # Initialise a dictionary to store the data for each target variable
    data_dict = {var: [] for var in target_variables}
    dates = []

    for datafile in datafiles:
        # Read data into a pd.Dataframe
        df = Plotter.unpickle(datafile)

        # If Dataframe contains data prepare for calculations
        check, df = prepare_dataframe(datafile, df)
        if check:
            # Process data for each target variable
            for var in target_variables:
                if var == 'OLR':
                    result = get_outgoing_longwave_radiation(plotter, df)
                elif var == 'Ice Fraction':
                    result = get_ice_fraction(df)
                # Append to dictionary
                data_dict[var].append(result)
            # Append the date for this file to the dates list
            dates.append(df['Datetime'].dt.date.iloc[0])
        else:
            # If DataFrame is empty, append NaN values
            for var in target_variables:
                data_dict[var].append(np.nan)
            # Append the date for this file to the dates list
            dates.append(np.nan)

    # Prepare and save the data for each target variable
    for var, results in data_dict.items():
        # Create a DataFrame from the results and dates
        df_to_save = pd.DataFrame({'Date': pd.to_datetime(dates), var: results})
        
        # Ensure results are numeric, converting non-numeric to NaN
        df_to_save[var] = pd.to_numeric(df_to_save[var], errors='coerce')
        
        # Save the DataFrame as a CSV for easier handling (you could also use .to_pickle for binary format)
        df_to_save.to_csv(f"{plotter.datapath}daily_{var.lower().replace(' ', '_')}.csv", index=False)


def load_and_sort_data(file_path, var):
    """
    Loads and sorts data from a .npy file.

    Parameters:
    - file_path (str): Path to the .npy file.
    - column_name (str): Name of the column for the data values (default is 'Value').

    Returns:
    - df_sorted (pd.DataFrame): DataFrame with sorted data by date.
    """
    data = pd.read_csv(file_path)
    df = pd.DataFrame(data, columns=['Date', var])
    df['Date'] = pd.to_datetime(df['Date'])
    df_sorted = df.sort_values(by='Date')
    return df_sorted

def add_grey_box(ax, df):
    """
    Adds grey boxes to the plot for every other year.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to add the grey boxes to.
    - df (pd.DataFrame): DataFrame with 'Year' column.
    """
    unique_years = sorted(df['Year'].unique())
    for i, year in enumerate(unique_years):
        if i % 2 == 0:
            ax.axvspan(i-0.5, i+0.5, color='grey', alpha=0.2)

def plot_statistical_timeseries(plotter, target_variables: List[str]):
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
            ylim = [0, 100]
        elif var == 'Ice Fraction':
            file_path = f"{plotter.datapath}daily_ice_fraction.csv"
            ylim = [0, 1]
        df = load_and_sort_data(file_path, var)

        # Ensure 'Date' is set as the DataFrame index
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        
        # Drop NaN values for plotting
        df = df.dropna(subset=[var])

        # Filter for only March, April, and May and add 'Year', 'Month', and 'Year-Month' columns
        df['Year'] = df.index.year
        df['Month'] = df.index.month_name().str[:3]
        df['Year-Month'] = df.index.strftime('%Y-%m')
        df_spring_months = df[df.index.month.isin([3, 4, 5])]

        # Create a subplot layout
        fig, ax = plt.subplots(figsize=(12, 6))

        # Violin Plot with Colors: visualises the distribution of data values for each spring month across years
        sns.violinplot(x='Year', y=var, hue='Month', data=df_spring_months, ax=ax, palette="muted", split=False)

        # Strip Plot: adds individual data points to the violin plot for detailed data visualization
        sns.stripplot(x='Year', y=var, hue='Month', data=df_spring_months, ax=ax, palette='dark:k', size=3, jitter=False, dodge=True)

        # Add grey box for visual separation of every other year for enhanced readability
        add_grey_box(ax, df_spring_months)

        # Customizing the plot with titles and labels
        ax.set_title(f"MAM Average {var}")
        ax.set_xlabel('Year')
        ax.set_ylabel(var)
        ax.set_ylim(ylim)
        ax.grid(axis='y', linestyle=':', color='k')
        ax.tick_params(axis='both', labelsize=plotter.fontsize)

        # Handling the legend to ensure clarity in distinguishing between different months
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:3], labels[:3], title='Month')

        # Save the plot to a file and close the plotting context to free up memory
        plt.tight_layout()
        plt.savefig(f"{plotter.datapath}daily_{var.lower().replace(' ', '_')}.png", dpi=300)
        plt.close()


def plot_pisco():
    """
    """
    # The path to the directory that contains the data files
    # datapath = "C:\\Users\\padra\\Documents\\Research\\data\\iasi\\2016"
    # datapath = "D:\\Data\\iasi\\"
    datapath = "/data/pdonnelly/iasi/metopb/"

    # Define temporal range to plot
    target_year = [2013]#, 2014, 2015, 2016, 2017, 2018, 2019]
    target_month = [3, 4, 5]
    target_days = [day for day in range(1, 32)] # Search all days in each month

    # Define plotting parameters
    fontsize = 10
    dpi = 300

    # Instantiate the Plotter and organise files
    plotter = Plotter(datapath, target_year, target_month, target_days, fontsize, dpi)
    
    # Define second-order target variables to calculate and plot
    target_variables=['OLR', 'Ice Fraction']

    # Plot data
    gather_daily_statistics(plotter, target_variables)
    plot_statistical_timeseries(plotter, target_variables)

if __name__ == "__main__":
    plot_pisco()