import gzip
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
import imageio
import numpy as np
import pandas as pd
import pickle
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class Plotter:
    """
    Class to contain useful plotting functions for the IASI dataset
    """
    def __init__(self, datapath: str, target_years: list, target_months: list, target_days: list, fontsize: float, dpi: int):
        """
        Initialises the Plotter class with a given data path.

        Args:
            datapath (str): The path to the data directory.
        """
        self.datapath = datapath
        self.target_years = target_years
        self.target_months = target_months
        self.target_days = target_days
        self.fontsize = fontsize
        self.dpi = dpi
        self.files_by_date: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.day_night_dictionary = {"night": 0, "day": 1, "twilight": 2}
        self.cloud_phase_dictionary = {"liquid": 1, "icy": 2, "mixed": 3, "clear": 4}


    # IASI-specific methods
    def _get_iasi_spectral_grid(self):
        spectral_grid = np.loadtxt("./inputs/iasi_spectral_grid.txt")
        channels = spectral_grid[:, 0]
        wavenumber_grid = spectral_grid[:, 1]
        return wavenumber_grid

    def get_dataframe_spectral_grid(self, df: pd.DataFrame) -> List[float]:
        # Get the full IASI spectral grid
        wavenumber_grid = self._get_iasi_spectral_grid()
        # Extract the numbers from the column names
        spectral_channels = df[[col for col in df.columns if 'Spectrum' in col]]
        channel_positions = spectral_channels.columns.str.split().str[-1].astype(int)
        # Extract the wavenumbers corresponding to the channel positions
        extracted_wavenumbers = [wavenumber_grid[position] for position in channel_positions]
        return extracted_wavenumbers


    # File I/O methods
    def _format_filepath_from_target_date_range(self) -> None:
        # Format years as 'YYYY'
        self.target_years = [str(year) for year in self.target_years]
        # Format months as 'mm' with leading zero if necessary
        self.target_months = [f"{month:02d}" for month in self.target_months]
        # Format days as 'dd' with leading zero if necessary
        self.target_days = [f"{day:02d}" for day in self.target_days]
        return
    
    def organise_files_by_date(self) -> None:
        """
        Organises .pkl.gz files in the data directory by date.

        The date is inferred from the directory structure: year/month/day.
        The result is stored in self.files_by_date, which is a dictionary
        mapping from (year, month, day) tuples to lists of file paths.

        This creates a dictionary with keys as dates (year, month, day) and values as lists of files.
        """
        self._format_filepath_from_target_date_range()

        for root, dirs, files in os.walk(self.datapath):
            for file in files:
                if ".pkl.gz" in file:
                    # Split the root directory path and get year, month and day
                    dir_structure = os.path.normpath(root).split(os.sep)
                    year, month, day = dir_structure[-3], dir_structure[-2], dir_structure[-1]

                    # Append the file path to the corresponding date
                    self.files_by_date[(year, month, day)].append(os.path.join(root, file))
    
    def select_files(self) -> List[str]:
        """
        Selects files from the dictionary created by organise_files_by_date method
        based on a target year, month, days and file name part.

        Args:
            target_years (str): The target year as a string.
            target_months (str): The target month as a string.
            target_days (List[str]): The target days as a list of strings.
            target_file_part (Optional[str]): The target part of the file name to select (defaults to None, the file containing all measurements)

        Returns:
            List[str]: List of the file paths that matched the conditions.
        """
        selected_files = []

        # Iterate through dictionary keys
        for (year, month, day), files in self.files_by_date.items():
            # Check if the year, month and day match your conditions
            if (year in self.target_years) and (month in self.target_months) and (day in self.target_days):
                # Iterate through the files for this date
                for file in files:
                    # Select file containing all measurements
                    selected_files.append(file)
        return sorted(selected_files)
    
    @staticmethod
    def unpickle(file):
        print(file)
        with gzip.open(file, 'rb') as f:
            df = pickle.load(f)
        return df

    # DataFrame manipulation methods
    @staticmethod
    def check_df(df: pd.DataFrame, local_time: Optional[str] = None, phase: Optional[str] = None, required_columns: Optional[List[str]] = None) -> bool:
        # Ensure the dataframe is not empty
        if df.empty:
            print(f"DataFrame empty")
            # if local_time:
            #     print(f"\n    No data available for time: {local_time}")
            # elif phase:
            #     print(f"\n    No data available for phase: {phase}")       
            return False     
            
        if required_columns:
            # Check for the presence of all required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing column(s) in DataFrame: {', '.join(missing_columns)}")
                return False
        else:
            return True

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


    # Plot-related methods
    @staticmethod
    def get_color_from_cmap(cmap):
        cmap = plt.get_cmap(cmap)
        return cmap(0.75)
    
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



class Geographic:
    def __init__(self, fontsize: int = 10):
        self.fontsize = fontsize

    def create_basemap(self, ax, lon_range: tuple, lat_range: tuple, resolution: str = "l"):
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
    def plot_geographical_histogram_2d(data: pd.DataFrame, lon_range: tuple, lat_range: tuple, m: Basemap, cmap: str = 'cividis', grid: float = 1.0):
        """
        Function to plot a two-dimensional histogram (histogram_2d) on a Basemap object in the specified longitude and latitude ranges.
        
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



class Spectrum():
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
        self.all_residuals: np.array = None
        self.histogram_2d: np.array = None
        self.x_positions: np.array = None
        self.y_positions: np.array = None

    def get_axis_ranges(self) -> None:
        # Set up wavenumber and spectrum ranges
        self.wavenumber_range = tuple((self.wavenumbers[0], self.wavenumbers[-1]))
        max_extent = self.data.abs().max().max()  
        self.spectrum_range = tuple((-1, 1)) #tuple((-max_extent, max_extent))
        return
    
    def get_bins(self) -> None:
        self.wavenumber_bins = np.arange(self.wavenumber_range[0], self.wavenumber_range[1] + (self.wavenumber_grid / 10), self.wavenumber_grid)
        self.spectrum_bins = np.arange(self.spectrum_range[0], self.spectrum_range[1] + (self.spectrum_grid / 10), self.spectrum_grid)
        return
    
    def build(self) -> None:        
        # Set up wavenumber and data ranges
        self.get_axis_ranges()

        # Generate bins for wavenumbers and data
        self.get_bins()
        return
    

    def _transform_histogram_2d_into_scatter(self) -> None:
        """
        Transform histogram into a scatter plot data.

        Parameters:
        H (np.ndarray): 2D histogram.
        xedges (np.ndarray): The bin edges along the first dimension of H.
        yedges (np.ndarray): The bin edges along the second dimension of H.

        Returns:
        Tuple containing the normalised histogram and the x and y centres.
        """
        xcentres = (self.x_positions[:-1] + self.x_positions[1:]) / 2
        ycentres = (self.y_positions[:-1] + self.y_positions[1:]) / 2

        self.histogram_2d = self.histogram_2d.flatten()
        self.x_positions = np.repeat(xcentres, len(ycentres))
        self.y_positions = np.tile(ycentres, len(self.x_positions)-1)

        self.histogram_2d = self.histogram_2d / self.histogram_2d.max()
        return
    
    def compute_single_histogram_2d(self) -> None:
        """
        Compute a 2D histogram for a single spectrum.

        Returns:
        Tuple containing the histogram, the xedges and yedges.
        """
        # Compute 2D histogram
        self.histogram_2d, self.x_positions, self.y_positions = np.histogram2d(self.wavenumbers, self.data.values.flatten(), bins=[self.wavenumber_bins, self.spectrum_bins])
        self.histogram_2d = np.nan_to_num(self.histogram_2d)
        return
    
    def compute_mean_histogram_2d(self) -> None:
        """
        Compute a 2D histogram for mean spectrum.

        Returns:
        Tuple containing the average histogram, the xedges and yedges.
        """
        # Calculate all residuals and normalised residuals at once
        residuals = np.subtract(self.data, self.spectrum_mean)
        normalised_residuals = np.divide(residuals, self.spectrum_mean)
        
        # Initialise histogram accumulator
        total_histogram = np.zeros((len(self.wavenumber_bins) - 1, len(self.spectrum_bins) - 1))

        # Calculate 2D histogram for each spectrum and accumulate
        for i in range(normalised_residuals.shape[0]):
            histogram, xedges, yedges = np.histogram2d(self.wavenumbers, normalised_residuals.iloc[i], density=True, bins=[self.wavenumber_bins, self.spectrum_bins])
            total_histogram += histogram

        # Flatten normalised residuals into 1D list
        self.all_residuals = normalised_residuals.values.flatten().tolist()

        # Average accumulated histogram and replace NaN values with zero
        self.histogram_2d = np.divide(total_histogram, self.data.shape[0])
        self.histogram_2d = np.nan_to_num(self.histogram_2d)

        # Set x and y positions for bins
        self.x_positions = xedges
        self.y_positions = yedges
        return


    def plot_spectrum(self, ax: plt.Axes, phase: Optional[str], color: str = 'black'):
        # Convert DataFrame to numpy array
        data_array = self.data.values

        # Calculate the maximum and minimum values for each wavelength
        max_values = np.max(data_array, axis=0)
        min_values = np.min(data_array, axis=0)

        # Plot mean spectrum and range
        ax.plot(self.wavenumbers, self.spectrum_mean, color=color, lw=1, label=phase)
        ax.fill_between(self.wavenumbers, min_values, max_values, color=color, lw=1, alpha=0.2)
        ax.set_xlim(self.wavenumber_range)
        ax.set_ylim((0, 1.01))
        ax.legend(loc='upper right')
        return ax

    def plot_histogram2d_in_histogram1d(self, ax: plt.Axes, color: str = 'color') -> plt.Axes:
        """
        Plot a one-dimensional histogram from all counts in histogram_2d.
        
        Parameters:
        ax (matplotlib.axes.Axes): Axes object on which histogram will be plotted.
        color (str): String name of the desired built-in color, default 'black'.

        Returns:
        ax (matplotlib.axes.Axes): The same Axes object with the plotted histogram_2d.
        """
        # Flatten the 2D histogram into a 1D array
        histogram = self.histogram_2d.flatten()

        # Calculate the range of values in the histogram
        hmin, hmax = np.min(histogram), np.max(histogram)
        print(hmin, hmax)
        # Plot 1D histogram of 2D histogram values
        ax.hist(histogram, bins=25, color=color, alpha=0.5, range=(hmin, hmax))
        # ax.set_xlim((-2, 2))
        # # ax.set_ylim((0, 251))
        return ax

    def plot_residuals_histogram_1d(self, ax: plt.Axes, number_of_bins: int = 20, color: str = 'black') -> plt.Axes:
        """
        Plot a one-dimensional histogram from all y-values collected when creating the 2-D heatmap.

        Parameters:
        ax (matplotlib.axes.Axes): Axes object on which histogram will be plotted.
        color (str): String name of the desired built-in color, default 'black'.

        Returns:
        ax (matplotlib.axes.Axes): The same Axes object with the plotted histogram.
        """
        # Convert to lower precision data type
        downsampled_residuals = np.asarray(self.all_residuals).astype(np.float16)

        # Plot 1D histogram of all normalised and downsampled residuals
        ax.hist(downsampled_residuals, bins=number_of_bins, density=True, range=self.spectrum_range, color=color, alpha=0.5)
        ax.set_xlim(self.spectrum_range)
        ax.set_ylim((0, 5))
        return ax

    def plot_histogram_2d(self, ax: plt.Axes, mode: str = '', cmap: str = 'cividis') -> plt.Axes:
        """
        Plot a two-dimensional histogram of a set of spectra.
        
        Parameters:
        ax (matplotlib.axes.Axes): Axes object on which histogram will be plotted.
        mode (str): String containing instructions on the kind of histogram to be plotted.
        cmap (str): String name of the desired built-in colormap, default 'cividis'.

        Returns:
        ax (matplotlib.axes.Axes): The same Axes object with the plotted histogram_2d.
        """
        if "scatter" in mode:
            self._transform_histogram_into_scatter()
            ax.scatter(self.x_positions, self.y_positions, c=self.histogram_2d, s=5, cmap=cmap, alpha=self.histogram_2d)
        else:
            X, Y = np.meshgrid(self.x_positions, self.y_positions)
            ax.pcolormesh(X, Y, self.histogram_2d.T, cmap=cmap) 

        ax.set_xlim(self.wavenumber_range)
        ax.set_ylim(self.spectrum_range)
        return ax