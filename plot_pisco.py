import os
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec

from pisco import Plotter, Spectrum, Geographic

def plot_spatial_distribution_scatter(datapath: str):
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
    # Use Plotter to organise files
    plotter.organize_files_by_date()

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
        # Initialize a new figure for the plot with three subplots
        fig = plt.figure(figsize=(10, 8), dpi=dpi)
        axes = gridspec.GridSpec(3, 2, figure=fig).subplots()
        fig.suptitle(f"IASI Spectra in the North Atlantic: {plotter.target_year}-{plotter.target_month}-{plotter.target_days[ifile]}", fontsize=fontsize+5, y=0.95)
        
        # Get current file and load data
        df = pd.read_csv(datafile, usecols=['Longitude', 'Latitude', 'Cloud Phase 1', 'Day Night Qualifier'])

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


def main():
    """
    """
    # The path to the directory that contains the data files
    datapath = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\iasi\\2019"

    # Define temporal range to plot
    target_year = '2019'
    target_month = '01'
    target_days = [str(day).zfill(2) for day in range(1, 32)]

    # Define plotting parameters
    fontsize = 8
    dpi = 150

    # Instantiate the Plotter and organise files
    plotter = Plotter(datapath, target_year, target_month, target_days, fontsize, dpi)

    # Plot data
    # plot_spatial_distribution_2Dhist(plotter)
    plot_spectral_distributon(plotter)

if __name__ == "__main__":
    main()
