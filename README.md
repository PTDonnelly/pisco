# Package for IASI Spectra and Cloud Observations (PISCO)

Pisco is a Python package designed to facilitate the extraction, processing and analysis of Infrared Atmospheric Sounding Interferometer (IASI) spectra and retrieved cloud products.

## Features

- Extracts data from raw binary files using optimised C scripts developed by the IASI team.
- Processes data into conveniently-formatted spatio-temporal data of IASI products: Level 1C calibrated spectra or Level 2 cloud products.
- Scans through a year-month-day range, extract cloud products (user-specified), extract and filter out co-incident spectra, and return a single csv for that day. Cloud products are currently Aqueous, Icy, Mixed, and Cloud-free (separated by filename).
- Supports correlation between Level 1C spectra and Level 2 cloud products.
- Configured by a separate JSON file, which is read and sets class attributes for the Configurer.

## Future

- Day/night separation of filtered spectra
- HDF5 output format if files tend to be too large

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


The main functionality of the package is provided through the `Extractor` class (configured by the `Configurer` class). The Extractor object uses these configurations to access, extract and process raw binary files of IASI data. (N.B.: a future version of Pisco will de-couple the directory and file paths from the Extractor module so that there is greater cohesion between the other modules.)

A default example is found in the module-level code `run_pisco.py`:

Remember to replace "path_to_your_config_file.json" with the actual path to your JSON configuration file if it's located somewhere else.

It scans through each specified year, month, and day in the configuration. For each day, it will either process Level 1C or Level 2 data, or correlate both Level 1C and Level 2 data, depending on your settings in the configuration file.

The `process_l1c()`, `process_l2()`, and `correlate_l1c_l2()` are functions imported from the `pisco.process_iasi` module, and they accept the Extractor class as an argument. Make sure these scripts are available and properly defined in your project. 

## Summary 

The Pisco project is designed to facilitate the efficient extraction and processing of cloud product data from the Infrared Atmospheric Sounding Interferometer (IASI) for spectrum analysis, along with correlating L1C spectra with L2 cloud products.

The main components of the Pisco project include the `Configurer`, `Extractor`, `L1CProcessor`, `L2Processor`, and `Correlator` classes.

1. **Configurer Class**: This class manages the setting of configuration parameters required for data extraction and processing. These parameters include date ranges, data level, data paths, IASI L1C target products, IASI channel indices, spatial ranges for binning, and desired cloud phase from L2 products. **Change these values for each analysis.**

2. **Extractor Class**: This class is the key driver of the data extraction process. It utilises the `Configurer` class to set the parameters for the extraction process and manages the flow of extraction and processing.

3. **L1CProcessor Class**: This class handles the Level 1C (L1C) IASI data product, i.e., the calibrated spectra. It provides methods for loading, processing, saving, and preparing the L1C data for correlation with L2 products.

4. **L2Processor Class**: This class handles the the Level 2 (L2) IASI data product, specifically cloud products. It offers methods for loading, processing, saving the L2 data, and preparing it for correlation with L1C products.

5. **Correlator Class**: This class handles correlating the L1C spectra with L2 cloud products. It achieves this by taking the resulting files from `L1CProcessor` and `L2Processor` and providing methods for correlating the data between these products. It looks for observations with a specific cloud phase and finds the corresponding spectra, then deletes the products of `L1CProcessor` and `L2Processor`.

The `main` function executes the entire process. It starts by creating an instance of the `Config` class to set the necessary parameters for the data extraction. Then, an `Extractor` is instantiated with this configuration, running the data extraction and processing sequence. The extractor iterates through the specified date ranges, extracting and processing the data accordingly. `L1CProcessor` uses a C script (OBR tool designed by IASI team) to read binary data, extracts variables to an intermediate binary file, extracts that to CSV or HDF5, and deletes the intermediate file. `L2Processor` uses a C script (unofficial BUFR reader designed by IASI team) to read binary data, and save to CSV (there is no intermediate step with this reader). The `Correlator` class then correlates the processed L1C and L2 data.

The `correlate` and `data_level` attributes of the `Config` class can be set in the JSON file to process each data level. `correlate` can be set True or False. False is non-destructive, it extracts binary data and saves spectra (L1C) and cloud products (L2) in CSV format. If True then the previous functionality can still apply, but when the spectra are filtered based on the outputs of the cloud products, a final CSV file is produced and the initial products are deleted. `correlate` can be True if there are existing L1C and L2 files (no need to run the Extractor of Processors) as logn as the file naming is consistent. By default, all are set to True as it is assumed that on a given day the user wants to simply generate a reduced dataset of spectra containing the desired cloud_phase.

It is crucial that process_l2() occurs before process_l1c(): the code first looks for the desired cloud products on the given day, if they are found it continues to extract spectra for the same day, then filter those sepctra to precise simulataneous measurements of the cloud products (i.e. spectra from pixels containing ice clouds). If none are found, it exists before executing process_l1c().


## Auxiliary scripts to extract binary data

`./bin/` contains the relevant binary executables of the IASI readers 
- `obr_v4` for Level 1C data: developed by the IASI team. Instructions can be found here https://www7.obs-mip.fr/wp-content-aeris/uploads/sites/12/2019/01/ETH-IASI-MU-696-CNe1r3.pdf, and in the L1CProcessor method `_build_command()`.
- `BUFR_iasi_clp_reader_from20190514` for Level 2 data: expands OBR tool to output reduced data directly to CSV (hence the asymmetry in the Processor classes). The conventions are as follows: 

To use on the SPIRIT cluster:
`./BUFR_iasi_clp_reader_from20190514 path_to_inputfile/inputfilename path_to_outputfile/outputfilename`

To obtain the output file:
`./BUFR_iasi_clp_reader_from20190514 /bdd/metopc/l2/iasi/2022/03/24/clp/W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPB+IASI_C_EUMP_20220324000252_49359_eps_o_clp_l2.bin clp_20220324000252.out`

The columns of the output file contain:
`latitude`, `longitude`, `date.time`, `orbit_number`, `scanline_number`, `pixel_number`, for 3 cloud formations: `cloud cover as percentage of pixel area`, `cloud top temperature`, `cloud top pressure`, `cloud phase`. Information on the latter is in the L1C_L2_Correlator method `_get_cloud_phase()`, it is the focus of this code for now.
