# Package for IASI Spectra and Cloud Observations (PISCO)

`Pisco` is a Python package designed to facilitate the extraction, processing and analysis of Infrared Atmospheric Sounding Interferometer (IASI) spectra and retrieved cloud products.

## Features

- Extracts IASI data from raw binary files on SPIRIT using optimised C script developed by the IASI team.
- Processes data into conveniently-formatted spatio-temporal data of IASI products: Level 1C (L1C) calibrated spectra and/or Level 2 (L2) cloud products.
- Scans through a year-month-day range, extract cloud products (user-specified), extract and filter out co-incident spectra.
- From this produces three files:
    - A single csv containing all IASI spectra (spectrally-integrated) in the geographic region for a given year/month/day
    - The same file in compresssed python pickle format (.pkl.gz)
    - The same as the csv file but spatially averaged onto a coarser one-degree lat-lon grid (for comparison to ERA5 atmospheric fields)
- Configured by a separate JSON file, which is read and sets class attributes for the Configurer() class.

## Installation

To install the package, clone this repository to your local machine by using the following command:

```bash
git clone https://github.com/PTDonnelly/pisco.git
```

## Usage Instructions
The pisco run is configured using the `/pisco/inputs/config.json` and executed simply by running the module-level script `/pisco/pisco.py`, which iterates over days in the specified time range and generates a SLURm job submission script for each date.

### To run `pisco` as-is:

#### JSON Configuration File
The configuration file is a JSON file that allows the user to specify parameters such as processing mode, year, month, day, output data path, and more. The JSON keys correspond to attributes of the `Configurer` class and are directly assigned during class instantiation. Here is a description of the parameters in the JSON configuration file:

- **submit_job (bool):** Specifies if the run should be submitted to SLURM (True) or run on a login node (False).
- **L1C (bool):** Specifies whether Level 1C data should be processed.
- **L2 (bool):** Specifies whether Level 2 data should be processed.
- **process (bool):** Specifies whether Level 1C and Level 2 data should be correlated for filtering spectra.
- **channels_mode (str):** Specifies the mode of selection for IASI spectral channels.
- **products (str):** Contains desired IASI L2 products, separated by commas (e.g., "clp,ozo,twt").
- **satellite_identifier (str):** The IASI satellite to be analyzed ("metopa" | "metopb" | "metopc").
- **quality_flags (str):** Three Yes/No flags describing three levels of IASI data quality ("yyy" = only good data, "nnn" = all data).
- **datapath (str):** User-defined data path for the processed output files.
- **delete_intermediate_files (bool):** Specifies whether intermediate OBR files and files from the Preprocessor should be deleted, leaving only the final merged dataset.
- **year_list (List[int]):** List of years for extraction.
- **month_list (List[int] or str):** List of specific months or "all" to scan all calendar months in a year.
- **day_list (List[int] or str):** List of specific days or "all" to scan all calendar days in a month.
- **months_in_year (List[int]):** Calendar months in each year.
- **days_in_months (List[int]):** Number of calendar days in each month.
- **latitude_range (List[int]):** The spatial range for binning latitudes [min, max].
- **longitude_range (List[int]):** The spatial range for binning longitudes [min, max].

There are three important parameters that control the execution, the `L1C`, `L2`, and `process` attributes of the `Config` class are set directly in the JSON file. If either or both `L1C`, `L2` are set to True (`true` with a lowercase "t" in JSON syntax), binary data are extracted and spectra (L1C) and/or cloud products (L2) are saved to CSV format. If `process` is set to True, this functionality still applies, with additional filtering of spectra based on the outputs of the cloud products, and production of multiple CSV files separated by day/night and cloud phase, and the initial products are deleted. `process` can be True if there are existing L1C and L2 files (no need to run the Extractor of Processors) as long as the file naming is consistent (set in the `Processor._get_intermediate_analysis_data_paths()` method). By default, all are set to True as it is assumed that on a given day the user wants to simply generate a reduced dataset of spectra with their corresponding clear sky/cloud phase indicator.

#### Executing the code

If you have set your parameters (satellite of interest, time range etc and you just want to run `pisco`, bearing in mind that it is designed to co-located spectra and cloud products by default) then simply run `/pisco/pisco.py`. It iterats through each specified year, month, and day in the configuration. For each day, it will process either L1C and L2 data (or both), then optionally correlate both L1C and L2 data, depending on your settings in the configuration file. For each day, it does a few things:

**Stage 1:** Instantiate an `Extractor` (which is initialised by the `Configurer`). The `Extractor` uses these configurations to prepare the extraction of the raw binary files of L1C and L2 IASI data, as well as defining file paths that will be used throughout the run. Depending on the configuration the raw files of either or both L1C and L2 data are read and stored in termporary files to be further reduced with the `Preprocessor`.

N.B. For L1C files the `obr_v4` executable is used (the official reader for spectra and a number of L2 products, documentation can be found here https://www7.obs-mip.fr/wp-content-aeris/uploads/sites/12/2019/01/ETH-IASI-MU-696-CNe1r3.pdf). For L2 files the `clpall_ascii` executable is used. `clpall` refers to the fact that it handles the reduction of cloud products throughout the operational lifetime of IASI important as the data standards changed multiple times. OBR does not currently support cloud products beyond 16/03/2019. You may need to run `chmod +x bin/obr_v4` and `chmod +x bin/clpall_ascii` in the root directory of the repository to enable executable permissions on the C executables.

**Stage 2:** Preprocessing of intermediate IASI data. The `Preprocessor` reads in the temporary files from above and structures them into pandas DataFrames for tidying and future manipulation. At this step, we add a Datetime (formatted as a Datetime object) and a day/night indicator based on local time (this already exists somewhere in the IASI products but it was straightforward enough to implement this here). `Preprocessor` supports handling large datasets by chunking and efficiently manages memory usage during the process. After these modifications, the Dataframes are saved to csv (L1C and L2 still seaprate for now) and the temporary files are deleted.

**Stage 3:** Processing and merging of L1C and L2 data. The `Processor` compares the preprocessed IASI products files and stores all spectra co-located with instances of a given coud phase. It loads, merges, and filters the preprocesssed L1C and L2 data, reduces the technical and scientific data fields to the most relevant ones for analysis (much of the instrument operational data is irrelevant), then saves the merged products to csv.

### To use the configuration architecture in your own code:
The `Configurer` module is designed to handle user-specified settings for the processing of IASI satellite data.

**Step 1:** Import the Module (by default this is done when initialising the Extractor class)
```python
from .configurer import Configurer
```

**Step 2:** Create a Configuration Object
Use your own JSON configuration file to initialize a `Configurer` object.

```python
config = Configurer("path_to_your_config_file.json")
```

## Optional Extra: Postprocessor Class
**This is essentially just plotting code used for this specific analysis. This class can be modified or ignored entirely.**

Generation of second-order products from the products of Processor is done with the Postprocessor. The class methods here are generalised for any target variables desired, dictionaries will dynamically populate based on inputs and self-organise into the correct columns in each variable output CSV file. Adding methods to Postprocessor can be done following the scheme in get_phase_fraction() and get_outgoing_longwave_radiation(), it is straightforward to add class attributes in addition to cloud_phase_names for instance-level access for multiple methods. Just remember to keep type hinting consistent! Define new methods and add them to Postprocessor.process_target_variables() as below (remember to also add your variable of interest to the target_variables list in the main() level function in ./scripts/gather_daily_statistics.py).

    # Define custom data processing method
    def get_test_values(self):
        # Initialize an empty dictionary to store values.
        # This is comparable to the "OLR" outer dictionary of data_dict
        test_variable = {}

        # Define a dictionary 'test_categories' with category names as keys and integers as values.
        # This dictionary represents the predefined categories and their associated numeric values.
        # This is comparable to the self.cloud_phase_names inner dictionaries of data_dict
        test_categories = {'A': 1, 'B': 2, 'C': 3}

        # Iterate over each category and store values
        # 'key' will be the category name (e.g., 'A', 'B', 'C'), and 'value' will be the associated integer (e.g., 1, 2, 3).
        # This is comparable to the 'Unknown', 'Water', 'Ice' columns etc. and their calculated values
        for key, value in test_categories.items():
            # Assign each category and its associated value from 'test_categories' to the 'test_variable' dictionary.
            # After this loop, 'test_variable' will have the same content as 'test_categories'.
            test_variable[key] = value

        # Return the 'test_variable' dictionary, which now contains the test values organized by category.
        return test_variable

    # Add it to the rest
    def process_target_variables(self, *args) -> None:
        """
        """
        for var in target_variables:

        if var == 'Test':
            values = self.get_test_values()
