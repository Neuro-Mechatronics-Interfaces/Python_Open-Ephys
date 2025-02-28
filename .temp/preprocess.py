"""
 Preprocess script (1/3) used for the gesture classifier. It preprocesses data files in a specified directory. Every file is associated with a single gesture

 Author: Jonathan Shulgach
 Last Modified: 2/25/2025
"""

import argparse
from pathlib import Path
import ephys_utilities.emg_processing as emg_proc
from ephys_utilities.ephys_utilities import OpenEphysClient

def main():
    parser = argparse.ArgumentParser(description="Modular Preprocessing of Open Ephys Data")
    parser.add_argument('--config_path', type=str, default='config.txt', help='Path to the config file containing the directory of .oebin files.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logs.')
    args = parser.parse_args()

    # Suppose your config.txt has a root_directory, so parse it:
    # Step 1: Get all .rhd file paths in the directory
    cfg = emg_proc.read_config_file(args.config_path)
    root_dir = Path(cfg.get('root_directory'))

    # Show all folders in the root_dir
    session_dir_list =  [str(i) for i in root_dir.iterdir()]
    for session_dir in session_dir_list:
        #print(session_dir)
        #filename = Path(file).name
        stop_channel_loop = False  # Reset the flag for each file
        start_times = []  # Clear start times for each file

        # Reads the session information and creates a dict with the EMG data, sampling rate, digital events, etc.
        result = emg_proc.read_ephys_file(session_dir_list[0])
        if not result:
            print("No data found in the file. Skipping...")
            continue

        emg_data = result['amplifier_data']
        time_vector = result['t_amplifier']  # Time vector for the data
        channel_names = [ch['custom_channel_name'] for ch in result['amplifier_channels']]
        sampling_rate = result['frequency_parameters']['amplifier_sample_rate']  # Sampling rate of the data
        do_manual = False if trigger_channel is not None else True

        # Step 3: Apply processing to EMG data (filtering, CAR, RMS, etc.)
        print(f"Processing file: {filename}")
        emg_data = emg_proc.notch_filter(emg_data, sampling_rate, 60)
        filt_data = emg_proc.filter_emg(emg_data, filter_type='bandpass', lowcut=30, highcut=500, fs=sampling_rate,
                                        verbose=True)
        car_data = emg_proc.common_average_reference(filt_data, True)
        grid_data = emg_proc.compute_grid_average(car_data, 8, 0)
        grid_ch = list(range(grid_data.shape[0]))
        rms_data = emg_proc.window_rms(car_data, window_size=800, verbose=True)

        if self.trigger_channel is not None:
            if 'board_dig_in_data' not in result:
                print("No digital input data found. Switching to manual...")
                do_manual = True

    # Load the first session
    session = Session(session_dir_list[0])
    print(session)


    # Recursively find all .rhd files
    #file_paths = list(root_dir.rglob('*.rhd'))
    #print(f"Found {len(file_paths)} .rhd files")



if __name__ == "__main__":
    main()