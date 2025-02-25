import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class RealtimePlotter:
    def __init__(self, client, sampling_rate=10000.0, plotting_interval=1.0):
        """
        :param client: An instance of OpenEphysClient (already created).
        :param sampling_rate: The sample rate for the data (Hz) for x-axis scaling.
        :param plotting_interval: Width of the x-axis in seconds (scroll window or full refresh).
        """
        self.client = client
        self.sampling_rate = sampling_rate
        self.plotting_interval = plotting_interval

        # We'll accumulate samples in self.ydata until we decide to update the plot
        self.ydata = np.array([], dtype=np.float32)

        # We'll increment self.frame_count, and when it hits self.frame_max, we'll update
        self.frame_count = 0
        # Each chunk of data might represent 213 samples, 1000 samples, etc.
        self.samples_per_fetch = 213

        # We'll set how many "fetches" to do before updating the plot
        self.fetches_per_frame = 5  # e.g. fetch 5 times, then update
        self.frame_max = self.fetches_per_frame

        # Set up the figure and the slider
        self.init_plot()

    def init_plot(self):
        """Initialize the matplotlib figure, axes, slider, line, etc."""
        self.figure, self.ax = plt.subplots()
        self.ax.set_facecolor('#001230')
        plt.subplots_adjust(left=0.1, bottom=0.2)  # leave room for slider

        # Initialize the line object
        self.hl, = self.ax.plot([], [], color='#d92eab', linewidth=0.5)

        # X-axis: from 0 to plotting_interval (in seconds)
        self.ax.set_xlim(0, self.plotting_interval)

        # Initial y-limits
        self.ylim0 = 200
        self.ax.set_ylim(-self.ylim0, self.ylim0)

        # Create slider for y-limits
        axcolor = 'lightgoldenrodyellow'
        axylim = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)
        self.sylim = Slider(axylim, 'Ylim', 1, 1000, valinit=self.ylim0)

        # When the slider moves, update y-limits
        def update_slider(val):
            yl = self.sylim.val
            self.ax.set_ylim(-yl, yl)
            plt.draw()

        self.sylim.on_changed(update_slider)

        # Create a timer to periodically fetch data and update
        self.timer = self.figure.canvas.new_timer(interval=50)  # 50 ms
        self.timer.add_callback(self.timer_callback)
        self.timer.start()

    def timer_callback(self):
        """
        This runs every 50 ms. Fetch data from the client, append to our buffer,
        and periodically refresh the plot.
        """
        # 1. Fetch new samples from channel 0 (adjust as needed)
        new_samples = self.client.get_samples(channel=0, n_samples=self.samples_per_fetch)

        # 2. Append to the local buffer
        if new_samples:
            self.ydata = np.concatenate([self.ydata, np.array(new_samples, dtype=np.float32)])

        # 3. Increment frame_count; update plot only every 'frame_max' fetches
        self.frame_count += 1
        if self.frame_count >= self.frame_max:
            # Compute time array in milliseconds (for example) or seconds
            # We'll plot in seconds, from 0 up to len(self.ydata)/sampling_rate
            if len(self.ydata) > 0:
                x = np.arange(len(self.ydata)) / self.sampling_rate  # in seconds

                # Update the line data
                self.hl.set_xdata(x)
                self.hl.set_ydata(self.ydata)

                # Keep x-limits fixed from 0 to self.plotting_interval
                # If you want a scrolling window, you could do:
                # if x[-1] > self.plotting_interval:
                #     start_x = x[-1] - self.plotting_interval
                #     self.ax.set_xlim(start_x, x[-1])
                # else:
                #     self.ax.set_xlim(0, self.plotting_interval)

                # For now, let's just set a fixed 0..plotting_interval
                self.ax.set_xlim(0, self.plotting_interval)

                # Let autoscale handle the y-range if you want dynamic scaling
                # self.ax.relim()
                # self.ax.autoscale_view(True, True, True)

                self.figure.canvas.draw()
                self.figure.canvas.flush_events()

            # Reset
            self.frame_count = 0
            # Clear out our buffer so we don't keep growing forever
            # Or keep it if you want to accumulate multiple frames:
            self.ydata = np.array([], dtype=np.float32)

    def run(self):
        """
        Display the plot and let the timer callback do its job.
        This call blocks until the user closes the plot window.
        """
        print("Starting the matplotlib event loop...")
        plt.show(block=True)
