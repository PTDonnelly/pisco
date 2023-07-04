import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import imageio
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap

class Plotter:
    """
    Class to contain useful plotting functions for the IASI dataset
    """
    def __init__(self, datapath: str):
        """
        Initializes the Plotter class with a given data path.

        Args:
            datapath (str): The path to the data directory.
        """
        self.datapath = datapath
        self.files_by_date: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    def get_iasi_spectral_grid(self):
        spectral_grid = np.loadtxt("./inputs/iasi_spectral_grid.txt")
        channels = spectral_grid[:, 0]
        wavenumber_grid = spectral_grid[:, 1]
        return channels, wavenumber_grid

    def iasi_channels_to_wavenumbers(self):
        pass

    def organize_files_by_date(self) -> None:
        """
        Organizes .csv files in the data directory by date.

        The date is inferred from the directory structure: year/month/day.
        The result is stored in self.files_by_date, which is a dictionary
        mapping from (year, month, day) tuples to lists of file paths.

        This creates a dictionary with keys as dates (year, month, day) and values as lists of files.
        """
        for root, dirs, files in os.walk(self.datapath):
            for file in files:
                if ".csv" in file:
                    # Split the root directory path and get year, month and day
                    dir_structure = os.path.normpath(root).split(os.sep)
                    year, month, day = dir_structure[-3], dir_structure[-2], dir_structure[-1]

                    # Append the file path to the corresponding date
                    self.files_by_date[(year, month, day)].append(os.path.join(root, file))

    
    def select_files(self, target_year: str, target_month: str, target_days: List[str], target_file_part: Optional[str] = None) -> List[str]:
        """
        Selects files from the dictionary created by organize_files_by_date method
        based on a target year, month, days and file name part.

        Args:
            target_year (str): The target year as a string.
            target_month (str): The target month as a string.
            target_days (List[str]): The target days as a list of strings.
            target_file_part (Optional[str]): The target part of the file name to select (defaults to None, the file containing all measurements)

        Returns:
            List[str]: List of the file paths that matched the conditions.
        """
        selected_files = []

        # Iterate through dictionary keys
        for (year, month, day), files in self.files_by_date.items():
            # Check if the year, month and day match your conditions
            if year == target_year and month == target_month and day in target_days:
                # Iterate through the files for this date
                for file in files:
                    # Check if the file name contains the target part
                    if target_file_part == None:
                        # Select file containing all measurements
                        selected_files.append(file)
                    elif target_file_part in file:
                        # Select file containing specified measurements
                        selected_files.append(file)

        return selected_files
    
    @staticmethod
    def png_to_gif(gifname: str, png_files: List[str], fps: int = 1, delete_png_files: bool = True) -> None:
        """
        Converts a list of png images into a gif animation.

        Args:
            gifname (str): The filepath and filename of the output gif.
            png_files (List[str]): List of paths to the png files to be included in the gif.
            delete_png_files (bool): If True, deletes the png files after creating the gif. Defaults to True.
        """

        # Once all figures are saved, use imageio to create a gif from all the png files
        with imageio.get_writer(gifname, mode='I', fps=fps) as writer:
            for png_file in png_files:
                image = imageio.imread(png_file)
                writer.append_data(image)
        
        # Optionally delete all png files after gif creation
        if delete_png_files:
            for png_file in png_files:
                os.remove(png_file)


    def create_basemap(self, lon_range: tuple, lat_range: tuple, ax, fontsize: int, resolution: str = "l"):
        """
        Function to create a Basemap with specified longitude and latitude ranges, and draw coastlines, meridians and parallels.
        
        Parameters:
        lon_range (tuple): Tuple containing the minimum and maximum longitudes for the basemap.
        lat_range (tuple): Tuple containing the minimum and maximum latitudes for the basemap.
        ax (matplotlib.axes.Axes): Axes object to draw the basemap on.
        fontsize (int): Font size for the labels on the meridians and parallels.
        
        Returns:
        m (mpl_toolkits.basemap.Basemap): The created Basemap object.
        """
        m = Basemap(projection='cyl', resolution=resolution, llcrnrlon=lon_range[0], llcrnrlat=lat_range[0], urcrnrlon=lon_range[1]+0.1, urcrnrlat=lat_range[1], ax=ax)
        
        # Draw coastlines and country borders
        m.drawcoastlines()

        # Add spatial grid
        meridians = np.arange(lon_range[0], lon_range[1]+1, 10)
        parallels = np.arange(lat_range[0], lat_range[1]+1, 10)
        m.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.5, dashes=[1, 1], fontsize=fontsize)
        m.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.5, dashes=[1, 1], fontsize=fontsize)
        return m
    
    def plot_geographical_heatmap(self, m: Basemap, data: pd.DataFrame, lon_range: tuple, lat_range: tuple, cmap: str, histogram_resolution: int=1):
        """
        Function to plot a two-dimensional histogram (heatmap) on a Basemap object in the specified longitude and latitude ranges.
        
        Parameters:
        m (mpl_toolkits.basemap.Basemap):Basemap object.
        data (pd.DataFrame): pandas DataFrame containing data values read from a CSV file.
        lon_range (tuple): Tuple containing the minimum and maximum longitudes for the basemap.
        lat_range (tuple): Tuple containing the minimum and maximum latitudes for the basemap.
        cmap (str): String naem of the desired built-in colormap.
        
        Returns:
        m (mpl_toolkits.basemap.Basemap): The same Basemap object.
        """
        # Define bins
        lon_bins = np.arange(lon_range[0], lon_range[1]+1, histogram_resolution)
        lat_bins = np.arange(lat_range[0], lat_range[1]+1, histogram_resolution)

        H, xedges, yedges = np.histogram2d(data['Longitude'], data['Latitude'], bins=[lon_bins, lat_bins])
        Lon, Lat = np.meshgrid(xedges, yedges)
        x, y = m(Lon, Lat)
        m.pcolormesh(x, y, H.T, cmap=cmap)  # Transpose H to align with coordinate grid
        return m