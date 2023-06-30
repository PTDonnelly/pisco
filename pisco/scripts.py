from typing import S

from .extraction import Extractor
from .preprocessing import Preprocessor
from .processing import Processor

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
        p = Preprocessor(ex.intermediate_file, ex.data_level, ex.config.latitude_range, ex.config.longitude_range, ex.config.products)
        
        # Open binary file and extract metadata
        p.open_binary_file()

        # Read common IASI record fields and store to pandas DataFrame
        p.read_record_fields(p.metadata._get_iasi_common_record_fields())

        # Limit observations to specified spatial range
        valid_indices = p.flag_observations_to_keep(p.metadata._get_iasi_common_record_fields())

        # Closed binary file and extract metadata
        p.close_binary_file()

    return valid_indices

def preprocess_iasi(ex: Extractor, data_level: str):
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
        ex.config.channels = ex.config.set_channels("all")
    ex.data_level = data_level
    ex.get_datapaths()
    ex.extract_files()

    # If IASI data was successfully extracted
    if ex.intermediate_file_check:
        # Preprocess the data into pandas DataFrames
        p = Preprocessor(ex.intermediate_file, ex.data_level, ex.config.latitude_range, ex.config.longitude_range, ex.config.products)
        p.preprocess_files(ex.year, ex.month, ex.day)
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
    p.correlate_spectra_with_cloud_products()
    return


def plot_spatial_distribution(datapath: str):
    import os
    import pandas as pd
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import imageio  # you'll use this later for gif creation

    filepaths = sorted([os.path.join(root, file) for root, dirs, files in os.walk(datapath) for file in files if ".csv" in file])

    # Initialize a new figure for the plot
    plt.figure(figsize=(8, 8))

    # Create a basemap of the world
    m = Basemap(projection='cyl', llcrnrlon=-61, llcrnrlat=29, urcrnrlon=1, urcrnrlat=61)

    # Draw coastlines and country borders
    m.drawcoastlines()

    # Plotting parameters
    colors = cm.turbo(np.linspace(0, 1, len(filepaths)))

    png_files = []

    # Walk through the directory
    for i, (file, color) in enumerate(zip(filepaths, colors)):
        print(file)

        # Load the data from the file into a pandas DataFrame
        data = pd.read_csv(file)
        
        # Plot the observations on the map
        x, y = m(data['Longitude'].values, data['Latitude'].values)
        m.scatter(x, y, latlon=True, marker=".", s=0.2, color="red", alpha=0.15)
        
        # save figure
        png_file = f"{datapath}/spatial_distribution_{i}.png"
        plt.savefig(png_file, dpi=300, bbox_inches='tight')

        # Append filename to list of png files
        png_files.append(png_file)

    # Once all figures are saved, use imageio to create a gif from all the png files
    with imageio.get_writer(f"{datapath}/spatial_distribution.gif", mode='I', fps=1) as writer:
        for png_file in png_files:
            image = imageio.imread(png_file)
            writer.append_data(image)
            
    # Optionally delete all png files after gif creation
    for png_file in png_files:
        os.remove(png_file)


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
        print(file)

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
    plt.savefig(f"{datapath}/average_spectra.png", dpi=300, bbox_inches='tight')