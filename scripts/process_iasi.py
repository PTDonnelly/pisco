from pisco import Extractor, Processor

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
    # Instantiate a Processor class 
    processor = Processor(ex)

    # Check that both L1C and L2 data exist
    if processor.check_l1c_l2_data_exist():
        # Load IASI spectra and cloud products
        processor.load_data()

        # Merge DataFrames, dropping uncorrelated rows and unwanted columns
        processor.combine_datasets()

        # Save merged and filtered DataFrame to compressed pickle
        processor.save_merged_products()
    
    return
