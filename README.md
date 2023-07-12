# Package for IASI Spectra and Cloud Observations (PISCO)

`Pisco` is a Python package designed to facilitate the extraction, processing and analysis of Infrared Atmospheric Sounding Interferometer (IASI) spectra and retrieved cloud products.

## Features

- Extracts data from raw binary files using optimised C scripts developed by the IASI team.
- Processes data into conveniently-formatted spatio-temporal data of IASI products: Level 1C calibrated spectra or Level 2 cloud products.
- Scans through a year-month-day range, extract cloud products (user-specified), extract and filter out co-incident spectra, and return a single csv for that day. Cloud products are currently Aqueous, Icy, Mixed, and Cloud-free (separated by filename).
- Configured by a separate JSON file, which is read and sets class attributes for the Configurer() class.

## Future

- Parallelisation of run_pisco.py, this can be easily achieved with Python and multiple packages will be tested.
- HDF5 output format if files tend to be too large
- Think about implementing direct reading of binary structure or memory mapping with np.memmap() instead of np.fromfile(), for faster reading of values (for flagging and storing)

## Installation

To install the package, clone this repository to your local machine by using the following command:

```bash
git clone https://github.com/PTDonnelly/pisco.git
```


## Usage Instructions for the Configurer Module

The `Configurer` module is designed to handle user-specified settings for the processing of IASI satellite data.

### Step 1: Import the Module (by default this is done when initialising the Extractor class)
```python
from .configurer import Configurer
```

### Step 2: Create a Configuration Object

Use your own JSON configuration file to initialize a `Configurer` object.

```python
config = Configurer("path_to_your_config_file.json")
```

### JSON Configuration File

The configuration file is a JSON file that allows the user to specify parameters such as processing mode, year, month, day, output data path, and many more. The JSON keys correspond to attributes of the `Configurer` class and they are directly assigned on class instantiation. Here is a description of the parameters in the JSON configuration file:

- **correlate: (bool)**, Specifies whether Level 1C and Level 2 data should be correlated for filtering spectra
- **L1C (bool):** Specifies whether Level 1C data should be processed.
- **L2 (bool):** Specifies whether Level 2 data should be processed.
- **year_list (List[int]):** List of years for extraction.
- **month_list (List[int]):** List of months for extraction.
- **day_list (List[int] or str):** List of specific days or "all" to scan all calendar days in the month.
- **days_in_months (List[int]):** Number of calendar days in each month.
- **datapath_out (str):** The user-defined data path for the processed output files.
- **targets (List[str]):** List of target IASI L1C products.
- **latitude_range (List[int]):** The spatial range for binning latitudes.
- **longitude_range (List[int]):** The spatial range for binning longitudes.
- **cloud_phase (int):** The desired cloud phase from the L2 products.

### Methods

The `Configurer` class has a post-processing method to set further attributes and perform checks:

- **set_channels():** This method sets the list of IASI channel indices. It returns a list with integers from 1 to n (where n is set to 5 in the provided code).



## Usage


The main functionality of the package is provided through the `Extractor` class (configured by the `Configurer` class). The Extractor object uses these configurations to access, extract and process raw binary files of IASI data. (N.B.: a future version of `Pisco` will de-couple the directory and file paths from the Extractor module so that there is greater cohesion between the other modules.)

A default example is found in the module-level code `run_pisco.py`:

Remember to replace "path_to_your_config_file.json" with the actual path to your JSON configuration file if it's located somewhere else.

It scans through each specified year, month, and day in the configuration. For each day, it will either process Level 1C or Level 2 data, or correlate both Level 1C and Level 2 data, depending on your settings in the configuration file.

N.B You may need to run `chmod +x bin/obr_v4` in the root directory of the repository to enable executable permissions on the OBR executable.

## Summary 

`Pisco` is designed to facilitate the efficient extraction and processing of cloud product data from the Infrared Atmospheric Sounding Interferometer (IASI) for spectrum analysis, along with correlating L1C spectra with L2 cloud products.

The main components of `Pisco` include the `Configurer`, `Extractor`, `Preprocessor`, and `Processor` classes.

1. **Configurer Class**: This class manages the setting of configuration parameters required for data extraction and processing. These parameters include date ranges, data level, data paths, IASI L1C target products, IASI channel indices, spatial ranges for binning, and desired cloud phase from L2 products. **Change these values for each analysis.**

2. **Extractor Class**: This class is the key driver of the data extraction process. It utilises the `Configurer` class to set the parameters for the extraction process and manages the flow of extraction and processing.

3. **Preprocessor Class**: This class handles the Level 1C (L1C) and Level 2 (L2) IASI data products, (calibrated spectra and retirieved products, respectively). It provides methods for loading, processing, saving, and preparing for analysis.

5. **Processor Class**: This class handles correlating the L1C spectra with L2 cloud products. It takes the resulting files from `Preprocessor` and correlates the spectral observations with a specific cloud phase, saves the corresponding spectra to a reduced dataset, then deletes the products of `Preprocessor`.

The `main` function executes the entire process. It starts by creating an instance of the `Extractor` class to set the necessary parameters for the data extraction. Immediately a `Config` class is instantiated with the configuration read from the `config.jsonc` file. The extractor iterates through the specified date ranges, extracting and processing the data accordingly. `Preprocessor` uses a C script (OBR tool designed by IASI team) to read binary data, extracts variables to an intermediate binary file, extracts that to CSV or HDF5, and deletes the intermediate binary file. The `Processor` class then correlates the processed L1C and L2 data.

There are three important parameters that control the execution, the `L1C`, `L2`, and `process` attributes of the `Config` class are set directly in the JSON file. If either or both `L1C`, `L2` are set to True (`true` with a lowercase "t" in JSON syntax), binary data are extracted and spectra (L1C) and/or cloud products (L2) are saved to CSV format. If `process` is set to True, this functionality still applies, with additional filtering of spectra based on the outputs of the cloud products, and production of multiple CSV files separated by day/night and cloud phase, and the initial products are deleted. `process` can be True if there are existing L1C and L2 files (no need to run the Extractor of Processors) as long as the file naming is consistent (set in the `Processor._get_intermediate_analysis_data_paths()` method). By default, all are set to True as it is assumed that on a given day the user wants to simply generate a reduced dataset of spectra containing the desired `cloud_phase`.

## Auxiliary scripts to extract binary data

`./bin/` contains the relevant binary executables of the IASI readers 
- `obr_v4` for Level 1C data: developed by the IASI team. Instructions can be found here https://www7.obs-mip.fr/wp-content-aeris/uploads/sites/12/2019/01/ETH-IASI-MU-696-CNe1r3.pdf, and in the `Preprocessor` method `_build_command()`. N.B. For now the OBR v4 does not work on L2 products, the newer version will soon be integrated into `pisco` for full funcitonality.