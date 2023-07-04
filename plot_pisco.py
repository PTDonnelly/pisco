from pisco import scripts

def main():
    """
    """
    # The path to the directory that contains the data files
    datapath = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\iasi\\2019"
    
    # Plot map
    scripts.plot_spatial_distribution_2Dhist(datapath)
    # plot_spectra(datapath)

if __name__ == "__main__":
    main()
