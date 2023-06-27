from .extraction import Extractor
from .preprocessing import Preprocessor
from .processing import Processor

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
    ex.data_level = data_level
    ex.get_datapaths()
    ex.extract_files()

    # If IASI data was successfully extracted
    if ex.intermediate_file_check:
        # Preprocess the data into pandas DataFrames
        p = Preprocessor(ex.intermediate_file, ex.data_level, ex.config.latitude_range, ex.config.longitude_range)
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
    
    filepaths = [os.path.join(root, file) for root, dirs, files in os.walk(datapath) for file in files if ".csv" in file]

    # Initialize a new figure for the plot
    plt.figure(figsize=(8, 8))

    # Create a basemap of the world
    m = Basemap(projection='cyl')

    # Draw coastlines and country borders
    m.drawcoastlines()
    # m.drawcountries()

    # Plotting parameters
    colors = cm.turbo(np.linspace(0, 1, len(filepaths)))
    
    # Walk through the directory
    for file, color in zip(filepaths, colors):
        print(file)

        # Load the data from the file into a pandas DataFrame
        data = pd.read_csv(file)
        # Plot the observations on the map
        x, y = m(data['Longitude'].values, data['Latitude'].values)
        m.scatter(x, y, latlon=True, marker=".", s=1, color=color)

    # Show the plot
    plt.savefig(f"{datapath}/spatial_distribution.png", dpi=300, bbox_inches='tight')

def plot_spectra(datapath: str):
    import os
    import pandas as pd
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np

    filepaths = [os.path.join(root, file) for root, dirs, files in os.walk(datapath) for file in files if ".csv" in file]

    # Plotting parameters
    colors = cm.turbo(np.linspace(0, 1, len(filepaths)))
    
    # Initialize a new figure for the plot
    plt.figure(figsize=(8, 8))

    # Walk through the directory
    for file, color in zip(filepaths, colors):
        print(file)

        # Load the data from the file into a pandas DataFrame
        df = pd.read_csv(file)
        
        # Get the spectrum (all columns with "Channel" in the name)
        spectrum_columns = [col for col in df.columns if "Channel" in col]
        spectrum = df[spectrum_columns[6:]].mean()
        
        channels = np.arange(len(spectrum))

        # Plot the average spectrum
        plt.plot(channels, spectrum, color=color, lw=0.5)
    
    plt.xlabel('Channel')
    plt.ylabel('Average intensity')
    # plt.yscale('log') 

    # Show the plot
    plt.savefig(f"{datapath}/average_spectra.png", dpi=300, bbox_inches='tight')