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
    pro = Processor(ex)

    # Check that both L1C and L2 data exist
    if pro.check_l1c_l2_data_exist():
        # Load IASI spectra and cloud products
        pro.load_data()      
        
        # Correlates measurements, keep matching locations and times of observation
        pro.correlate_datasets()
        
        # Merge DataFrames, dropping uncorrelated rows and unwanted columns
        pro.combine_datasets()

        # Save merged and filtered DataFrame to compressed pickle
        pro.save_merged_products()
    
    return
