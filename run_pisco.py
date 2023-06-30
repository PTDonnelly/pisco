from pisco import Extractor, flag_data, preprocess_iasi, process_iasi

def main():
    """PISCO: Package for IASI Spectra and Cloud Observations

    For each date specified, open raw binary files, reduce into intermediate files using optimised C scripts
    developed by IASI team, then produce conveniently-formatted spatio-temporal data
    of IASI products: L1C calibrated spectra or L2 cloud products.
    """
    # Point to location of jsonc configuration file
    path_to_config_file = "./inputs/config.jsonc"
    
    # Instantiate an Extractor class to get data from raw binary files
    ex = Extractor(path_to_config_file)
    
    # Scan years, months, days (specific days or all calendar days, dependent on Config attributes)
    for year in ex.config.year_list:
        ex.year = f"{year:04d}"

        month_range = ex.config.month_list if (not ex.config.month_list == "all") else range(1, 13)
        for im, month in enumerate(month_range):
            ex.month = f"{month:02d}"

            day_range = ex.config.day_list if (not ex.config.day_list == "all") else range(1, ex.config.days_in_months[im] + 1)
            for day in day_range:
                ex.day = f"{day:02d}"
                
                if (ex.config.L1C) or (ex.config.L2):
                    valid_indices = flag_data(ex, data_level="l1c")
                if ex.config.L1C:
                    preprocess_iasi(ex, valid_indices, data_level="l1c")
                if ex.config.L2:
                    preprocess_iasi(ex, valid_indices, data_level="l2")
                if ex.config.process:
                    process_iasi(ex)

if __name__ == "__main__":
    main()
