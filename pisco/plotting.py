import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import imageio
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

class Plotter:
    """
    Class to contain useful plotting functions for the IASI dataset
    """
    def __init__(self, datapath: str, target_year: str, target_month: str, target_days: List[str], target_file_part: Optional[str] = None):
        """
        Initializes the Plotter class with a given data path.

        Args:
            datapath (str): The path to the data directory.
        """
        self.datapath = datapath
        self.target_year = target_year
        self.target_month = target_month
        self.target_days = target_days
        self.files_by_date: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self.day_night_dictionary = {'night': 0, 'day': 1, 'twilight': 2}
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


    def plot_geographical_heatmap(self, data: pd.DataFrame, lon_range: tuple, lat_range: tuple, m: Basemap, cmap: str, histogram_resolution: int=1):
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
    
    def plot_geographical_scatter(self):
        pass


    def gather_dataframe_spectra(self, file_groups: dict, ifile: int, df_name_1='day_icy', df_name_2='night_icy'):
        df_1 = pd.read_csv(file_groups[df_name_1]['files'][ifile]).filter(regex='Spectrum ')
        df_2 = pd.read_csv(file_groups[df_name_2]['files'][ifile]).filter(regex='Spectrum ')
        merged_df = pd.concat([df_1, df_2], axis=0)
        return self.get_dataframe_spectral_grid(merged_df), merged_df
    
    def plot_spectra_by_cloud_phase(self, ax, file_groups, ifile, df_name_1, df_name_2, color):
        spectrum_wavenumbers, spectrum_merged_df = self.gather_dataframe_spectra(file_groups, ifile, df_name_1, df_name_2)
        spectrum_mean = spectrum_merged_df.mean(axis=0) * 1000
        spectrum_stddev = spectrum_merged_df.std(axis=0) * 1000
        ax.plot(spectrum_wavenumbers, spectrum_mean, color=color, lw=1)
        ax.fill_between(spectrum_wavenumbers, spectrum_mean-spectrum_stddev, spectrum_mean+spectrum_stddev, color=color, alpha=0.2)
        ax.set_xlim((spectrum_wavenumbers[0], spectrum_wavenumbers[-1]))
        return