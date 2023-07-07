import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
import imageio
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class Plotter:
    """
    Class to contain useful plotting functions for the IASI dataset
    """
    def __init__(self, datapath: str, target_year: str, target_month: str, target_days: List[str], fontsize: float, dpi: int):
        """
        Initializes the Plotter class with a given data path.

        Args:
            datapath (str): The path to the data directory.
        """
        self.datapath = datapath
        self.target_year = target_year
        self.target_month = target_month
        self.target_days = target_days
        self.fontsize = fontsize
        self.dpi = dpi
        self.files_by_date: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.day_night_dictionary = {"night": 0, "day": 1, "twilight": 2}
        self.cloud_phase_dictionary = {"liquid": 1, "icy": 2, "mixed": 3, "clear": 4}

    def _get_iasi_spectral_grid(self):
        spectral_grid = np.loadtxt("./inputs/iasi_spectral_grid.txt")
        channels = spectral_grid[:, 0]
        wavenumber_grid = spectral_grid[:, 1]
        return wavenumber_grid

    def get_dataframe_spectral_grid(self, df: pd.DataFrame) -> List[float]:
        # Get the full IASI spectral grid
        wavenumber_grid = self._get_iasi_spectral_grid()
        # Extract the numbers from the column names
        channel_positions = df.columns.str.split().str[-1].astype(int)
        # Extract the wavenumbers corresponding to the channel positions
        extracted_wavenumbers = [wavenumber_grid[position] for position in channel_positions]
        return extracted_wavenumbers


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
    
    def select_files(self) -> List[str]:
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
            if year == self.target_year and month == self.target_month and day in self.target_days:
                # Iterate through the files for this date
                for file in files:
                    # Select file containing all measurements
                    selected_files.append(file)

        return selected_files
    
    @staticmethod
    def check_df(datafile: str, sub_df: pd.DataFrame, local_time: Optional[str] = None, phase: Optional[str] = None) -> bool:
        # Ensure the dataframe is not empty
        if sub_df.empty:
            print(f"\DataFrame empty: {datafile}")
            if local_time:
                print(f"\n    No data available for time: {local_time}")
            if phase:
                print(f"\n    No data available for phase: {phase}")
            return False

    def extract_by_cloud_phase_and_day_night(self, df: pd.DataFrame, conditions_dict: dict = None):
        """
        Function to extract subset of DataFrame based on conditions for Cloud Phase and Day Night Qualifier.

        Args:
            df: DataFrame to process.
            conditions_dict: Dictionary where keys are day_night qualifiers and values are lists of cloud phases. 
                            Default is None, which returns all data.
        
        Returns:
            Subset of the original DataFrame or the original DataFrame if no conditions_dict is provided.
        """
        # Ensure the input is correct
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The 'df' input should be a pandas DataFrame.")
            
        if conditions_dict is not None and not isinstance(conditions_dict, dict):
            raise ValueError("The 'conditions_dict' input should be a dictionary.")
            
        if conditions_dict is None:
            return df
        else:
            df_filtered = pd.DataFrame()

            for day_night_str, cloud_phases_str in conditions_dict.items():
                day_night = self.day_night_dictionary[day_night_str]
                cloud_phases = [self.cloud_phase_dictionary[phase_str] for phase_str in cloud_phases_str]
                for cloud_phase in cloud_phases:
                    mask_cloud_phase = df['Cloud Phase 1'] == cloud_phase
                    mask_day_night = df['Day Night Qualifier'] == day_night
                    df_temp = df[mask_cloud_phase & mask_day_night]
                    df_filtered = pd.concat([df_filtered, df_temp])
            return df_filtered


    def finalise_plot(self, filename: str, ifile: int, png_files: List[str], dpi: int, hspace: float = 0.1, wspace: float = 0.1):
        # Final adjustments
        plt.subplots_adjust(hspace=hspace, wspace=wspace)

        # Save figure
        png_file = os.path.join(self.datapath, f"{filename}_{ifile}.png")
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
        plt.close()

        # Append filename to list of png files
        png_files.append(png_file)
        return png_files
    
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



    def create_basemap(self, lon_range: tuple, lat_range: tuple, ax, resolution: str = "l"):
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
        m.drawmeridians(meridians, labels=[0,0,0,1], linewidth=0.5, dashes=[1, 1], fontsize=self.fontsize)
        m.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.5, dashes=[1, 1], fontsize=self.fontsize)
        return m

    @staticmethod
    def plot_geographical_heatmap(data: pd.DataFrame, lon_range: tuple, lat_range: tuple, m: Basemap, cmap: str = 'cividis', grid: float = 1.0):
        """
        Function to plot a two-dimensional histogram (heatmap) on a Basemap object in the specified longitude and latitude ranges.
        
        Parameters:
        data (pd.DataFrame): pandas DataFrame containing data values read from a CSV file.
        lon_range (tuple): Tuple containing the minimum and maximum longitudes for the basemap.
        lat_range (tuple): Tuple containing the minimum and maximum latitudes for the basemap.
        m (mpl_toolkits.basemap.Basemap):Basemap object.
        cmap (str): String name of the desired built-in colormap.
        grid (float): float describing the spatial resolution of the histogram grid in degrees

        Returns:
        m (mpl_toolkits.basemap.Basemap): The same Basemap object.
        """
        # Define bins and compute histogram
        lon_bins = np.arange(lon_range[0], lon_range[1]+1, grid)
        lat_bins = np.arange(lat_range[0], lat_range[1]+1, grid)
        H, xedges, yedges = np.histogram2d(data['Longitude'], data['Latitude'], bins=[lon_bins, lat_bins])

        lon, lat = np.meshgrid(xedges, yedges)
        x, y = m(lon, lat)
        m.pcolormesh(x, y, H.T, cmap=cmap)  # Transpose H to align with coordinate grid
        return m
    
    @staticmethod
    def plot_geographical_contour(data: pd.DataFrame, lon_range: tuple, lat_range: tuple, m: Basemap, cmap: str = 'cividis', grid: float = 1.0, levels: int = 10):
        """
        Function to plot a two-dimensional histogram (contour) on a Basemap object in the specified longitude and latitude ranges.
        
        Parameters:
        data (pd.DataFrame): pandas DataFrame containing data values read from a CSV file.
        lon_range (tuple): Tuple containing the minimum and maximum longitudes for the basemap.
        lat_range (tuple): Tuple containing the minimum and maximum latitudes for the basemap.
        m (mpl_toolkits.basemap.Basemap):Basemap object.
        cmap (str): String name of the desired built-in colormap.
        grid (float): float describing the spatial resolution of the histogram grid in degrees

        Returns:
        m (mpl_toolkits.basemap.Basemap): The same Basemap object.
        """
        # Define bins and compute histogram
        lon_bins = np.arange(lon_range[0], lon_range[1]+1, grid)
        lat_bins = np.arange(lat_range[0], lat_range[1]+1, grid)
        H, xedges, yedges = np.histogram2d(data['Longitude'], data['Latitude'], bins=[lon_bins, lat_bins])

        # Apply a Gaussian filter to the histogram
        H = gaussian_filter(H, sigma=1)

        # Compute bin centres from edges
        xcentres = xedges[:-1] + 0.5 * (xedges[1:] - xedges[:-1])
        ycentres = yedges[:-1] + 0.5 * (yedges[1:] - yedges[:-1])

        lon, lat = np.meshgrid(xcentres, ycentres)
        x, y = m(lon, lat)
        
        # Create a filled contour plot
        m.contourf(x, y, H.T, levels=levels, cmap=cmap, antialiased=True)
        return m
    
    @staticmethod
    def plot_geographical_scatter(data: pd.DataFrame, m: Basemap, s: float = 1.0, alpha: float = 1.0, cmap: str = 'cividis'):
        """
        Function to plot a two-dimensional scatter on a Basemap object in the specified longitude and latitude ranges.
        
        Parameters:
        data (pd.DataFrame): pandas DataFrame containing data values read from a CSV file.
        m (mpl_toolkits.basemap.Basemap):Basemap object.
        s (float): float describing the markersize (as default functionality)
        alpha (float): float describing the marker opacity (as default functionality)
        cmap (str): String name of the desired built-in colormap.

        Returns:
        m (mpl_toolkits.basemap.Basemap): The same Basemap object.
        """
        # Plot the observations on the map
        x, y = m(data['Longitude'].values, data['Latitude'].values)
        m.scatter(x, y, latlon=True, marker=".", s=s, cmap=cmap, alpha=alpha)
        return m


class SpectralHeatmap():
    def __init__(self, data: pd.DataFrame, wavenumbers: List[float], wavenumber_grid: float = 5.0, spectrum_grid: float = 0.05):
        """
        Initialise SpectralHeatmap with data, wavenumbers and optional parameters w_grid and spectrum_grid.

        Parameters:
        data (pd.DataFrame): Data for plotting.
        wavenumbers (List[float]): List of wavenumbers.
        wavenumber_grid (float): Wavenumber grid size, default 5.0.
        spectrum_grid (float): Data grid size, default 0.05.
        """
        self.data = data
        self.spectrum_mean = data.mean(axis=0)
        self.wavenumbers = wavenumbers
        self.wavenumber_grid = wavenumber_grid
        self.spectrum_grid = spectrum_grid
        self.wavenumber_range: Tuple[float] = None
        self.spectrum_range: Tuple[float] = None
        self.wavenumber_bins: List[float] = None
        self.spectrum_bins: List[float] = None

    @staticmethod
    def transform_histogram_into_scatter(H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform histogram into a scatter plot data.

        Parameters:
        H (np.ndarray): 2D histogram.
        xedges (np.ndarray): The bin edges along the first dimension of H.
        yedges (np.ndarray): The bin edges along the second dimension of H.

        Returns:
        Tuple containing the normalised histogram and the x and y centres.
        """
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2

        H = H.flatten()
        xcentres = np.repeat(xcentres, len(ycentres))
        ycentres = np.tile(ycentres, len(xedges)-1)

        H_normalised = H / H.max()
        return H_normalised, xcentres, ycentres
    
    def compute_single_histogram(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute a 2D histogram for a single spectrum.

        Returns:
        Tuple containing the histogram, the xedges and yedges.
        """
        # Compute 2D histogram
        H, xedges, yedges = np.histogram2d(self.wavenumbers, self.data.values.flatten(), bins=[self.wavenumber_bins, self.spectrum_bins])
        H = np.nan_to_num(H)
        return H, xedges, yedges

    def compute_mean_histogram(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute a 2D histogram for mean spectrum.

        Returns:
        Tuple containing the average histogram, the xedges and yedges.
        """
        # Initialize histogram accumulator
        H_accum = np.zeros((len(self.wavenumber_bins) - 1, len(self.spectrum_bins) - 1))

        # Loop over data to accumulate histograms
        for _, spectrum in self.data.iterrows():
            residuals = np.subtract(spectrum.values, self.spectrum_mean)
            normalised_residuals = np.divide(residuals, self.spectrum_mean)
            H, xedges, yedges = np.histogram2d(self.wavenumbers, normalised_residuals, bins=[self.wavenumber_bins, self.spectrum_bins])
            H_accum += H

        # Average accumulated histogram and replace NaN values with zero
        H_average = np.divide(H_accum, self.data.shape[0])
        H_average = np.nan_to_num(H_average)
        return H_average, xedges, yedges

    def get_axis_ranges(self) -> None:
        # Set up wavenumber and spectrum ranges
        self.wavenumber_range = tuple((self.wavenumbers[0], self.wavenumbers[-1]))
        max_extent = self.data.abs().max().max()  
        self.spectrum_range = tuple((-max_extent, max_extent))
        return
    
    def get_bins(self) -> None:
        self.wavenumber_bins = np.arange(self.wavenumber_range[0], self.wavenumber_range[1] + (self.wavenumber_grid / 10), self.wavenumber_grid)
        self.spectrum_bins = np.arange(self.spectrum_range[0], self.spectrum_range[1] + (self.spectrum_grid / 10), self.spectrum_grid)
        return
    
    def construct_spectral_heatmap(self, mode: str) -> plt.Axes:
        """
        Build a two-dimensional histogram (heatmap) from spectral data. The value of the "mode" argument determines
        which instance methods to run to computer the desired histogram.
        
        Parameters:
        mode (str): String containing instructions on the kind of histogram to be plotted.

        Returns:
        histogram (np.Array): Numpy nd.array containing the 2-D histogram data
        x_positions (np.Array): Numpy nd.array containing the x_ positions (xedges for hsitogram and xcentres for scatter)
        y_positions (np.Array): Numpy nd.array containing the x_ positions (yedges for hsitogram and ycentres for scatter)
        """
        # Define a dictionary mapping keywords to the corresponding functions for computing histograms
        compute_histogram_funcs = {
            "single": self.compute_single_histogram,
            "mean": self.compute_mean_histogram,
        }
        
        # Define a dictionary mapping keywords to the corresponding functions for transforming histograms
        transform_histogram_funcs = {
            "scatter": self.transform_histogram_into_scatter,
            "default": lambda H, xedges, yedges: (H, xedges, yedges)
        }
        
        # Iterate over compute_histogram_funcs to find which function to use based on the mode
        for key in compute_histogram_funcs:
            if key in mode:
                compute_func = compute_histogram_funcs[key]  # Set the compute function based on the mode
                break
        else:
            # If the loop doesn't break, raise an error that the input mode doesn't contain any of the expected keywords
            raise ValueError(f"Plot mode == {mode} must contain 'mean' or 'single'")

        # Iterate over transform_histogram_funcs to find which function to use based on the mode
        for key in transform_histogram_funcs:
            if key in mode:
                transform_func = transform_histogram_funcs[key]  # Set the transform function based on the mode
                break
        else:
            # If the loop doesn't break set the transform function to the default transform function
            transform_func = transform_histogram_funcs["default"]

        # Call the compute function to get histogram data
        histogram, x_positions, y_positions = compute_func()

        # Apply the transformation to the histogram data and return the result
        return transform_func(histogram, x_positions, y_positions)

    def plot(self, ax: plt.Axes, mode: str, cmap: str = 'cividis') -> plt.Axes:
        """
        Plot a two-dimensional histogram (heatmap) on a Basemap object in the specified longitude and latitude ranges.
        
        Parameters:
        ax (matplotlib.axes.Axes): Axes object on which histogram will be plotted.
        mode (str): String containing instructions on the kind of histogram to be plotted.
        cmap (str): String name of the desired built-in colormap, default 'cividis'.

        Returns:
        ax (matplotlib.axes.Axes): The same Axes object with the plotted heatmap.
        """
        # Check if mode parameter is a string
        if not isinstance(mode, str):
            raise ValueError("Mode must be a string.")
        
        # Set up wavenumber and data ranges
        self.get_axis_ranges()

        # Generate bins for wavenumbers and data
        self.get_bins()

        # Build a two-dimensional histogram (heatmap) from spectral data.
        histogram, x_positions, y_positions = self.construct_spectral_heatmap(mode)
        
        # Define a dictionary mapping keywords to the corresponding functions for plotting histograms
        plot_histogram_funcs = {
            "scatter": lambda: ax.scatter(x_positions, y_positions, c=histogram, s=5, cmap=cmap, alpha=histogram),
            "default": lambda: ax.pcolormesh(*np.meshgrid(x_positions, y_positions), histogram.T, cmap=cmap) 
        }

        # Iterate over plot_histogram_funcs to find which function to use based on the mode
        for key in plot_histogram_funcs:
            if key in mode:
                plot_func = plot_histogram_funcs[key]  # Set the plot function based on the mode
                break
        else:
            # If the loop doesn't break set the transform function to the default transform function
            plot_func = plot_histogram_funcs["default"]
        
        plot_func()  # Call the selected plot function
        ax.set_xlim(self.wavenumber_range)
        ax.set_ylim(self.spectrum_range)
        return ax