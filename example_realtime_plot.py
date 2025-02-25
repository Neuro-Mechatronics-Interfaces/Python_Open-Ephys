import argparse
from ephys_utilities.ephys_utilities import OpenEphysClient
from ephys_utilities.realtime_plotter import RealtimePlotter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Print verbose output")
    args = parser.parse_args()

    # 1. Create OpenEphysClient instance
    client = OpenEphysClient()

    # 2. Create the real-time plotter, use the sample rate that matches your Open Ephys config
    plotter = RealtimePlotter(client, sampling_rate=10000.0, plotting_interval=10)

    # 3. Start the plotting (blocking call until user closes the window)
    plotter.run()

    print("Plot window closed. Exiting.")