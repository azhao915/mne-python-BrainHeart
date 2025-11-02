from PyQt5.QtWidgets import QWidget, QHBoxLayout
import pyqtgraph as pg
import numpy as np

from mne.brainheart.testing.test_spectrum import welch_with_CI

class TFRWidget(QWidget): 
    def __init__(
            self, 
            tf,
            curr_channel, 
            curr_time, 
            window_duration, 
            dB = False
    ): 
        super().__init__()
        self.data = tf.data
        self.times = tf.times
        self.freqs = tf.freqs
        self.sfreq_tf = tf.sfreq

        self.ch_names = tf.ch_names

        self.dB = dB

        # Current States
        self.curr_channel = curr_channel
        self.curr_time = curr_time
        self.window_duration = window_duration

        self._setup_ui()
        self.redraw()

    def _setup_ui(
            self
    ): 
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Time-Frequency Widget

        self.plot_tfr_widget = pg.PlotWidget()
        self.plot_tfr_widget.setLabel("left", "Frequency (Hz)")
        self.plot_tfr_widget.setLabel("bottom", "Time (s)")
        layout.addWidget(self.plot_tfr_widget, 5)

        self.image_item = pg.ImageItem()
        self.plot_tfr_widget.addItem(self.image_item)

        colormap = pg.colormap.get("inferno")
        self.image_item.setColorMap(colormap)

        # Power Spectrum Widget
        self.power_spectrum_widget = pg.PlotWidget()
        self.power_spectrum_widget.setLabel("left", "Frequency (Hz)")
        self.power_spectrum_widget.setLabel("bottom", "Power")
        layout.addWidget(self.power_spectrum_widget, 1)

        # Plotting elements for the Power Spectrum Widget
        self.power_spectrum_item = self.power_spectrum_widget.plot([], [])
        self.spectrum_lower = pg.PlotDataItem([], [])
        self.spectrum_upper = pg.PlotDataItem([], [])

        self.spectrum_fill = pg.FillBetweenItem(
            self.spectrum_lower, 
            self.spectrum_upper, 
            brush = pg.mkBrush(
                color = (255, 0, 0, 50)
            )
        )
        self.power_spectrum_widget.addItem(self.spectrum_fill)
        
        # Link the Frequencies in the two widgets
        self.power_spectrum_widget.setYLink(self.plot_tfr_widget)

        self._setup_tfr_range()

    
    def _update_display(
            self, 
            curr_channel_name: str | None = None,
            curr_channel: int | None = None,
            curr_time: float | None = None, 
            window_duration: float | None = None,
    ): 
        # Exact same as for the time plotting
        if curr_channel_name is not None: 
            if curr_channel_name in self.ch_names: 
                curr_channel = self.ch_names.index(curr_channel_name)
            else: 
                curr_channel = None
        if curr_channel is not None: 
            self.curr_channel = curr_channel
        if curr_time is not None: 
            self.curr_time = curr_time 
        if window_duration is not None: 
            self.window_duration = window_duration
        self.redraw()


    def redraw(self): 
        start_idx = int(self.curr_time * self.sfreq_tf)
        end_idx = int((self.curr_time + self.window_duration)*self.sfreq_tf)    
        end_idx = min(end_idx, len(self.times))

        time_slice = slice(start_idx, end_idx)
        tfr_data = self.data[self.curr_channel, :, time_slice]

        self.image_item.setImage(tfr_data.T, autoLevels = False)

        vmin = np.percentile(tfr_data, 1)
        vmax = np.percentile(tfr_data, 99)

        self.image_item.setLevels([vmin, vmax])

        # Now update the Power Spectrum
        _, psd, _, lower, upper = welch_with_CI(None, tfr_data)
        self.power_spectrum_item.setData(psd, self.freqs)
        self.spectrum_lower.setData(lower, self.freqs)
        self.spectrum_upper.setData(upper, self.freqs)

        self.image_item.setRect(
            self.curr_time, # x position (time) 
            self.freqs[0], # y position (freq)
            self.window_duration,  # width in time
            self.freqs[-1] - self.freqs[0] # height in freq
        )

        self.plot_tfr_widget.setXRange(
            self.curr_time, 
            self.curr_time + self.window_duration, 
            padding = 0
        )

    
    def _setup_tfr_range(self): 
        # (x, y) is bottom-left corner
        
        self.plot_tfr_widget.setYRange(
            self.freqs[0], 
            self.freqs[-1], 
            padding = 0
        )