from pisco import Plotter, scripts

def main():
    """
    """
    # The path to the directory that contains the data files
    datapath = "C:\\Users\\padra\\Documents\\Research\\projects\\contrails\\iasi\\2019"

    # Define temporal range to plot
    target_year = '2019'
    target_month = '01'
    target_days = [str(day).zfill(2) for day in range(1, 32)]

    # Instantiate the Plotter and organise files
    plotter = Plotter(datapath, target_year, target_month, target_days)

    # Plot map
    scripts.plot_spatial_distribution_2Dhist(plotter)
    # scripts.plot_spectral_distributon(plotter)

if __name__ == "__main__":
    main()
