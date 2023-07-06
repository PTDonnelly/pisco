from typing import List
import numpy as np

from .extraction import Extractor
from .preprocessing import Preprocessor
from .processing import Processor
from .plotting import Plotter

def flag_data(ex: Extractor, data_level: str):
    """This function is a truncated form of the main preprocess_iasi function, but bypasses most
    of the functionality. It creates a much smaller dataset with OBR, scans it and returns the indices 
    of observations within the specified latitude-longitude range.
    """
    
    # Use OBR to extract IASI data from raw binary files
    if data_level == "l1c":
        ex.config.channels = ex.config.set_channels("flag")
    ex.data_level = data_level
    ex.get_datapaths()
    ex.extract_files()

    # If IASI data was successfully extracted
    if ex.intermediate_file_check:
        # Preprocess the data into pandas DataFrames
        p = Preprocessor(ex.intermediate_file, ex.data_level, ex.config.latitude_range, ex.config.longitude_range)
        
        # Open binary file and extract metadata
        p.open_binary_file()

        # Limit observations to specified spatial range
        valid_indices = p.flag_observations_to_keep(p.metadata._get_iasi_common_record_fields())

        # Closed binary file and extract metadata
        p.close_binary_file()

        return valid_indices
    return


def preprocess_iasi(ex: Extractor, valid_indices: np.array, data_level: str):
    """
    This function is used to process IASI (Infrared Atmospheric Sounding Interferometer) data 
    by extracting raw binary files and preprocessing them into pandas DataFrames.

    It first uses an Extractor object to fetch the raw data files for a specific data level, 
    and then performs extraction. 

    If the extraction is successful, the function creates a Preprocessor object using the path 
    to the extracted intermediate file and the data level. This Preprocessor object preprocesses 
    the files to create a DataFrame for the given date.

    Parameters:
    ex (Extractor): An Extractor object which contains methods and attributes for data extraction.
    data_level (str): The data level string, which determines the level of data to be extracted 
                      and preprocessed.

    Returns:
    None: The function performs extraction and preprocessing operations but does not return a value.
    """
    # Use OBR to extract IASI data from raw binary files
    if data_level == "l1c":
        ex.config.channels = ex.config.set_channels("range")
    ex.data_level = data_level
    ex.get_datapaths()
    ex.extract_files()

    # If IASI data was successfully extracted
    if ex.intermediate_file_check:
        # Preprocess the data into pandas DataFrames
        p = Preprocessor(ex.intermediate_file, ex.data_level, ex.config.latitude_range, ex.config.longitude_range)
        p.preprocess_files(ex.year, ex.month, ex.day, valid_indices)
    return


def process_iasi(ex: Extractor):
    """
    Correlate level 1C spectra and level 2 cloud products.

    Compares processed IASI products from CSV files and
    stores all spectra co-located with instances of a given Cloud Phase.

    Args:
        Instance of the Extractor class 
    
    Result:
        A CSV file containing all spectra at those locations and times.
    """  
    p = Processor(ex.config.datapath_out, ex.year, ex.month, ex.day, ex.config.cloud_phase)
    p.merge_spectra_and_cloud_products()
    return



def plot_spatial_distribution_scatter(datapath: str):
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    # Instantiate the Plotter and organise files
    plotter = Plotter(datapath)
    plotter.organize_files_by_date()

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

        # Initialize a new figure for the plot
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
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.basemap import Basemap
    
    def check_sub_df(sub_df: pd.DataFrame, phase: str) -> None:
        # Ensure the dataframe is not empty
        if sub_df.empty:
            raise ValueError(f"No data available for phase: {phase}")
        else:
            return
        
    def plot_scatter(m: Basemap, file_groups, ifile):
        # Get all spectra
        all_files = plotter.select_files(target_year, target_month, target_days)
        all_data = pd.read_csv(all_files[ifile], usecols=['Longitude', 'Latitude'])
        # Get icy spectra
        day_icy_data = pd.read_csv(file_groups['day_icy']['files'][ifile], usecols=['Longitude', 'Latitude'])
        night_icy_data = pd.read_csv(file_groups['day_icy']['files'][ifile], usecols=['Longitude', 'Latitude'])

        # Plot the observations on the map
        x, y = m(all_data['Longitude'].values, all_data['Latitude'].values)
        m.scatter(x, y, latlon=True, marker=".", s=1, color="silver", alpha=0.8)
       
        # Plot the observations on the map
        x, y = m(day_icy_data['Longitude'].values, day_icy_data['Latitude'].values)
        m.scatter(x, y, latlon=True, marker=".", s=1.25, color="crimson")
        
        # Plot the observations on the map
        x, y = m(night_icy_data['Longitude'].values, night_icy_data['Latitude'].values)
        m.scatter(x, y, latlon=True, marker=".", s=1.25, color="crimson")

    def plot_spectrum(file_groups: dict, ifile: int, ax: object):        
        plotter.plot_spectra_by_cloud_phase(ax, file_groups, ifile, 'day_icy', 'night_icy', 'royalblue')
        plotter.plot_spectra_by_cloud_phase(ax, file_groups, ifile, 'day_liquid', 'night_liquid', 'forestgreen')
        plotter.plot_spectra_by_cloud_phase(ax, file_groups, ifile, 'day_mixed', 'night_mixed', 'darkorchid')        
        return
    
    # Use Plotter to organise files
    plotter.organize_files_by_date()

    # Select files in time range
    datafiles = plotter.select_files()
    
    # Define plotting parameters
    fontsize = 8
    dpi = 360
    png_files = []
    nx, ny = 3, 2
    
    titles = ['Spectrum', 'Normalised Residuals', 'Histogram-Residuals']
    phases = ['icy', 'liquid', 'mixed']
    colors = ['royalblue', 'forestgreen', 'darkorchid']

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
        # Initialize a new figure for the plot with three subplots
        fig = plt.figure(figsize=(15, 9), dpi=dpi)
        axes = gridspec.GridSpec(ny, nx, figure=fig).subplots()
        fig.suptitle(f"IASI Spectra in the North Atlantic: {plotter.target_year}-{plotter.target_month}-{plotter.target_days[ifile]}", fontsize=fontsize+5, y=0.95)
        
        # Get current file and load data
        df = pd.read_csv(datafile)

        for iax, (ax, (group, attrs)) in enumerate(zip(axes.flat, plot_params.items())):
            local_time = attrs["local_time"]
            phase = attrs["phase"]
            sub_df = plotter.extract_by_cloud_phase_and_day_night(df, {local_time: [phase]}).filter(regex='Spectrum ')

            # Make sure DataFrame is not empty
            check_sub_df(sub_df, phase)

            # Create a basemap of the world
            m = plotter.create_basemap(lon_range, lat_range, ax, fontsize)

            # Plot the observations on the map as a 2D histogram
            plotter.plot_geographical_heatmap(sub_df, lon_range, lat_range, m, attrs["cmap"])

    for ifile in range(len(target_days)):
        # Initialize a new figure for the plot with three subplots
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        gs = gridspec.GridSpec(3, 2, figure=fig)
        fig.suptitle(f"IASI Spectra in the North Atlantic: {target_year}-{target_month}-{target_days[ifile]}", fontsize=fontsize+5, y=0.95)

        axes = gs.subplots()  # Create the subplots from the GridSpec

        for iax, (ax, (group, attrs)) in enumerate(zip(axes.flat, file_groups.items())):
            if iax == 0:
                # Create a basemap of the world
                m = plotter.create_basemap(lon_range, lat_range, ax, fontsize)
                plot_scatter(m, file_groups, ifile)
            elif iax == 1:
                plot_spectrum(file_groups, ifile, ax)
                ax.set_xlabel(r'Wavenumber (cm$^{-1}$)', labelpad=1, fontsize=fontsize)
                ax.set_ylabel(r'Radiance ($mWm^{-2}srm^{-1}m$)', labelpad=1, fontsize=fontsize)
            elif iax > 1:
                # Create a basemap of the world
                m = plotter.create_basemap(lon_range, lat_range, ax, fontsize)
                # Get current file and load data
                file = attrs["files"][ifile]
                data = pd.read_csv(file, usecols=['Longitude', 'Latitude'])
                # Plot the observations on the map as a 2D histogram
                plotter.plot_geographical_heatmap(m, data, lon_range, lat_range, attrs["cmap"])

            # Add a title to the plot
            ax.set_title(attrs["title"], fontsize=fontsize+1)

        # Final adjustments
        plt.subplots_adjust(hspace=0.35, wspace=0.1)

        # Save figure
        png_file = f"{datapath}/2D_hist_{ifile}.png"
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
        plt.close()

        # Append filename to list of png files
        png_files.append(png_file)

    # Convert all individual pngs to animated gif
    plotter.png_to_gif(f"{datapath}/2D_hist.gif", png_files)


def plot_spatial_distribution_unity(datapath: str):
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    
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
            df = pd.read_csv(file[ifile], usecols=['Longitude', 'Latitude', 'Cloud Phase 1'])

            # Truncate latitude and longitude values onto a 10-degree grid
            df['Longitude_truncated'] = (df['Longitude'] // 10 * 10).astype(int)  # Group longitudes into 10 degree bins
            df['Latitude_truncated'] = (df['Latitude'] // 10 * 10).astype(int)  # Group latitudes into 10 degree bins

            df_grouped = df.groupby(['Longitude_truncated', 'Latitude_truncated', 'Cloud Phase 1']).size().reset_index(name='Counts')

            # Append the grouped dataframe to the overall dataframe
            df_all_grouped.append(df_grouped)
        return pd.concat(df_all_grouped, ignore_index=True)
    
    def plot_grouped_data(ax: object, df_all_grouped: pd.DataFrame, cloud_phase_colors: List[str]) -> None:
        # Iterate over unique combinations of Latitude and Longitude
        for (lat, lon), df_sub in df_all_grouped.groupby(['Latitude_truncated', 'Longitude_truncated']):

            # Compute normalized coordinates for the subplot within the main figure (note: this depends on your actual lat-lon ranges)
            # Assuming the longitude ranges from -180 to 180 and the latitude from -90 to 90
            normalized_lon = (lon + 60) / 60
            normalized_lat = (lat - 30) / 30
            
            # Create an inset_axes object that corresponds to a subplot within the grid cell
            ax_sub = ax.inset_axes([normalized_lon, normalized_lat, 1/6, 1/3])
            
            # Create a bar plot for each cloud phase within the group
            for phase in [1, 2, 3]:
                # Get the count for the current cloud phase
                count = df_sub.loc[df_sub['Cloud Phase 1'] == phase, 'Counts'] / df_sub['Counts'].sum()
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
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    def check_sub_df(sub_df: pd.DataFrame, phase: str) -> None:
        # Ensure the dataframe is not empty
        if sub_df.empty:
            raise ValueError(f"No data available for phase: {phase}")
        else:
            return
        
    def plot_spectrum(ax, spectrum_wavenumbers, spectrum_mean, spectrum_error, phase, color):
        ax.plot(spectrum_wavenumbers, spectrum_mean, color=color, lw=1, label=phase)
        ax.fill_between(spectrum_wavenumbers, spectrum_mean-spectrum_error, spectrum_mean+spectrum_error, color=color, alpha=0.2)
        ax.set_xlim((spectrum_wavenumbers[0], spectrum_wavenumbers[-1]))
        ax.set_ylim((0, 1.01))
        ax.legend(loc='upper right')

    def plot_residuals(ax, spectrum_wavenumbers, residuals, color):
        ax.plot(spectrum_wavenumbers, [0]*len(spectrum_wavenumbers), color=color, lw=1)
        ax.fill_between(spectrum_wavenumbers, -residuals, residuals, color=color, alpha=0.2)
        ax.set_xlim((spectrum_wavenumbers[0], spectrum_wavenumbers[-1]))
        ax.set_ylim((-0.4, 0.401))

    def plot_histogram(ax, residuals, color):
        ax.hist(residuals, bins=25, color=color, alpha=0.5)
        ax.set_xlim((0, 0.51))
        ax.set_ylim((0, 251))

    # Use Plotter to organise files
    plotter.organize_files_by_date()

    # Select files in time range
    datafiles = plotter.select_files()
    
    # Define plotting parameters
    fontsize = 8
    dpi = 360
    png_files = []
    titles = ['Spectrum', 'Normalised Residuals', 'Histogram-Residuals']
    phases = ['icy', 'liquid', 'mixed']
    colors = ['royalblue', 'forestgreen', 'darkorchid']

    for ifile, datafile in enumerate(datafiles):
        # Initialize a new figure for the plot with three subplots
        fig = plt.figure(figsize=(15, 9), dpi=dpi)
        gs = gridspec.GridSpec(3, 5, figure=fig)
        fig.suptitle(f"IASI Spectra in the North Atlantic: {plotter.target_year}-{plotter.target_month}-{plotter.target_days[ifile]}", fontsize=fontsize+5, y=0.95)
        
        # Get current file and load data
        df = pd.read_csv(datafile)

        for irow, (phase, color) in enumerate(zip(phases, colors)):
            sub_df = plotter.extract_by_cloud_phase_and_day_night(df, {'day': [phase], 'night': [phase]}).filter(regex='Spectrum ')

            # Make sure DataFrame is not empty
            check_sub_df(sub_df, phase)
            
            # Calculate plotting data
            spectrum_wavenumbers = plotter.get_dataframe_spectral_grid(sub_df)
            spectrum_mean = sub_df.mean(axis=0) * 1000
            spectrum_error = sub_df.std(axis=0) * 1000
            # spectrum_error = (sub_df.max(axis=0) - sub_df.min(axis=0)) * 1000
            residuals = spectrum_error / spectrum_mean

            for icol in range(nx):
                if icol == 0:
                    ax = fig.add_subplot(gs[irow, icol:icol+2])
                    plot_spectrum(ax, spectrum_wavenumbers, spectrum_mean, spectrum_error, phase, color)
                    xlabel = r'Wavenumber (cm$^{-1}$)'
                    ylabel = r'Radiance ($mWm^{-2}srm^{-1}m$)'
                elif icol == 2:
                    ax = fig.add_subplot(gs[irow, icol:icol+2])
                    plot_residuals(ax, spectrum_wavenumbers, residuals, color)
                    xlabel = r'Wavenumber (cm$^{-1}$)'
                    ylabel = r'Normalised Error'
                elif icol == 4:
                    ax = fig.add_subplot(gs[irow, icol])
                    plot_histogram(ax, residuals, color)
                    xlabel = r'Normalised Spread'
                    ylabel = r'Counts'

                if irow == 0:
                    if icol in [0, 1]:  # For the first two columns
                        ax.set_title(titles[0], fontsize=fontsize+1)
                    elif icol in [2, 3]:  # For the next two columns
                        ax.set_title(titles[1], fontsize=fontsize+1)
                    elif icol == 4:  # For the last column
                        ax.set_title(titles[2], fontsize=fontsize+1)
                if irow == 2:
                    ax.set_xlabel(xlabel, labelpad=1, fontsize=fontsize)
                ax.set_ylabel(ylabel, labelpad=1, fontsize=fontsize)

        # Save figure and store png filename for gif conversion
        png_file = os.path.join(plotter.datapath, f"spectral_distribution_{ifile}.png")
        plotter.finalise_plot(png_file, png_files, dpi, hspace=0.2, wspace=0.4)

    # Convert all individual pngs to animated gif
    plotter.png_to_gif(f"{plotter.datapath}/spectral_distribution.gif", png_files)


def plot_spectra(datapath: str):
    import os
    import pandas as pd
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np

    spectral_grid = np.loadtxt("./inputs/iasi_spectral_grid.txt")
    wavenumber_grid = spectral_grid[:, 1]

    filepaths = [os.path.join(root, file) for root, dirs, files in os.walk(datapath) for file in files if ".csv" in file]
    spectra = []
    # Plotting parameters
    colors = cm.turbo(np.linspace(0, 1, len(filepaths)))
    
    # Initialize a new figure for the plot
    plt.figure(figsize=(8, 4))

    # Walk through the directory
    for file, color in zip(filepaths, colors):
        # Load the data from the file into a pandas DataFrame
        df = pd.read_csv(file)
        
        # Get the spectrum (all columns with "Channel" in the name)
        spectrum_columns = [col for col in df.columns if "Channel" in col]
        spectra.append(df[spectrum_columns[6:]].mean())
    
    spectrum_mean = np.mean(np.array(spectra), axis=0)
    spectrum_stddev = np.std(np.array(spectra), axis=0)
        
    # Plot the average spectrum
    plt.plot(wavenumber_grid, spectrum_mean, color='k', lw=1)
    # plt.plot(wavenumber_grid, spectrum_mean+spectrum_stddev, color='k', lw=0.5)
    # plt.plot(wavenumber_grid, spectrum_mean-spectrum_stddev, color='k', lw=0.5)
    plt.xlabel(r'Wavenumber (cm$^{-1}$)')
    plt.ylabel(r'Radiance ($Wm^{-2}srm^{-1}m$)')
    # plt.yscale('log') 

    # Show the plot
    plt.savefig(f"{datapath}/average_spectra.png", dpi=540, bbox_inches='tight')