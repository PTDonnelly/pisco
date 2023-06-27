from pisco import plot_spatial_distribution, plot_spectra

def main():
    """
    """
    # The path to the directory that contains the data files
    datapath = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\iasi\\2020"
    
    # Plot map
    # plot_spatial_distribution(datapath)
    plot_spectra(datapath)

if __name__ == "__main__":
    main()
