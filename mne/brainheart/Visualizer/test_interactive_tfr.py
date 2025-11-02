from PyQt5.QtGui import QKeyEvent
import mne 
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, 
    QApplication, 
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QPushButton, 
    QLabel)

from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import sys

from ChannelManager import ChannelManager
from AnnotationsManager import AnnotationsManager

from mne.brainheart.testing.test_spectrum import welch_with_CI



class TFRBrowser(QMainWindow):
    def __init__(self, 
                 tf: mne.time_frequency.BaseTFR, 
                 raw: mne.io.BaseRaw | None = None,
                 annotations: mne.Annotations | None = None,
                 dB: float = False,
                 window_duration: float = 10.0, 
                 current_time: float = 0.0,
                 ):
        super().__init__()
        self.data = tf.data # (n_chan, n_freqs, n_times)
        self.dB = dB
        if self.dB: 
            self.data = 20*np.log10(self.data)
        self.times = tf.times
        self.sfreq_tf = tf.sfreq
        self.freqs = tf.freqs

        # If given, then have the raw trace, if not then ignore it
        self.raw = raw

        # Initialize the annotations if possible
        if annotations is None and raw is not None: 
            annotations = raw.annotations
        self.annotations = annotations
        self.display_annotations = True

        self.ch_names = tf.ch_names
        self.n_channels = len(self.ch_names)

        # Current State
        self.current_channel = 0
        self.window_duration = 10.0
        self.current_time = 0.0
        
        # Window parameters
        self.window_duration = window_duration  # seconds to show
        self.current_time = current_time  # start time

        # Annotations Manager
        self.annot_manager = AnnotationsManager(
            annotations = self.annotations,
            current_time = self.current_time, 
            window_duration = self.window_duration 
        )

        self.setWindowTitle("TFR Browser")
        self.resize(1200, 800)
        self._setup_ui()
        self._update_display()
        
    def _setup_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        self.channel_manager = ChannelManager(self.ch_names, self.current_channel)
        self.channel_manager.channel_changed.connect(self._on_channel_changed)

        main_layout.addWidget(self.channel_manager)
        
        if self.raw is not None: 
            # Time Trace
            self.plot_time_widget = pg.PlotWidget()
            self.plot_time_widget.setLabel("left", "V")
            self.plot_time_widget.setLabel("bottom", "Time (s)")

            main_layout.addWidget(self.plot_time_widget)

            self.line_item = self.plot_time_widget.plot([], [])

            self.annot_manager.register_plot(
                plot_name = "time", 
                plot_widget = self.plot_time_widget)


        tfr_layout = QHBoxLayout()
        main_layout.addLayout(tfr_layout)
        
        self.plot_tfr_widget = pg.PlotWidget()
        self.plot_tfr_widget.setLabel("left", "Frequency (Hz)")
        self.plot_tfr_widget.setLabel("bottom", "Time (s)")
        tfr_layout.addWidget(self.plot_tfr_widget, 5)


        # Link the Annotations Manager
        self.annot_manager.register_plot(
            plot_name = "time_frequency", 
            plot_widget = self.plot_tfr_widget
        )


        self.image_item = pg.ImageItem()
        self.plot_tfr_widget.addItem(self.image_item)

        colormap = pg.colormap.get("inferno")
        self.image_item.setColorMap(colormap)

        # Now add the Power Spectrum Widget
        self.plot_power_spectrum_widget = pg.PlotWidget()
        self.plot_power_spectrum_widget.setLabel("left", "Frequency (Hz)")
        self.plot_power_spectrum_widget.setLabel("bottom", "Power")
        tfr_layout.addWidget(self.plot_power_spectrum_widget, 1)

        self.power_spectrum_item = self.plot_power_spectrum_widget.plot([], [])

        self.spectrum_lower = pg.PlotDataItem([], [])
        self.spectrum_upper = pg.PlotDataItem([], [])

        self.spectrum_fill = pg.FillBetweenItem(
            self.spectrum_lower, 
            self.spectrum_upper, 
            brush = pg.mkBrush(color = (255, 0, 0, 50))
        )
        self.plot_power_spectrum_widget.addItem(self.spectrum_fill)
        
        #Link the X-axes so they zoom/pan together
        if self.raw is not None: 
            self.plot_time_widget.setXLink(self.plot_tfr_widget)


    def _on_channel_changed(self, new_channel): 
        self.current_channel = new_channel
        self._update_display()

    
    def _update_display(self): 
        start_idx = int(self.current_time * self.sfreq_tf)
        end_idx = int((self.current_time + self.window_duration)*self.sfreq_tf)    
        end_idx = min(end_idx, len(self.times))

        time_slice = slice(start_idx, end_idx)
        tfr_data = self.data[self.current_channel, :, time_slice]

        self.image_item.setImage(tfr_data.T, autoLevels = True)

        # Now update the Power Spectrum
        _, psd, _, lower, upper = welch_with_CI(None, tfr_data)
        self.power_spectrum_item.setData(psd, self.freqs)
        self.spectrum_lower.setData(lower, self.freqs)
        self.spectrum_upper.setData(upper, self.freqs)

        vmin = np.percentile(tfr_data, 1)
        vmax = np.percentile(tfr_data, 99)

        self.image_item.setLevels([vmin, vmax])

        if self.raw is not None: 
            # Update the upper trace
            start_idx_raw = int(self.current_time*raw.info["sfreq"])
            end_idx_raw = int((self.current_time + self.window_duration)*raw.info["sfreq"]) 
            times = self.current_time + np.arange(end_idx_raw - start_idx_raw)/raw.info["sfreq"]
            curr_index_in_raw = mne.pick_channels(raw.ch_names, [tf.ch_names[self.current_channel]])[0]
            raw_trace = raw.get_data(
                picks = curr_index_in_raw, 
                return_times = False, 
                start = start_idx_raw, 
                stop = end_idx_raw).flatten()
            self.line_item.setData(times, raw_trace)


        # (x, y) is bottom-left corner
        self.image_item.setRect(
            self.current_time, # x position (time) 
            self.freqs[0], # y position (freq)
            self.window_duration,  # width in time
            self.freqs[-1] - self.freqs[0] # height in freq
        )

        self.plot_tfr_widget.setXRange(
            self.current_time, 
            self.current_time + self.window_duration, 
            padding = 0
        )

        self.plot_tfr_widget.setYRange(
            self.freqs[0], 
            self.freqs[-1], 
            padding = 0
        )

        self.annot_manager.update_annotations(
            self.current_time, 
            self.window_duration)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None: 
            return
        if event.key() == Qt.Key_Right: 
            self.current_time += self.window_duration / 4
            self.current_time = min(self.current_time,
                                    self.times[-1] - self.window_duration)
            self._update_display()

        elif event.key() == Qt.Key_Left:
            self.current_time -= self.window_duration / 4
            self.current_time = max(0, self.current_time)
            self._update_display()
            
        elif event.key() == Qt.Key_Up:
            # Previous channel
            self.channel_manager._prev_channel()
            
        elif event.key() == Qt.Key_Down:
            # Next channel
            self.channel_manager._next_channel()
            
        elif event.key() == Qt.Key_Home:
            # Decrease window duration
            self.window_duration = max(1.0, self.window_duration * 0.8)
            self._update_display()
            
        elif event.key() == Qt.Key_End:
            # Increase window duration
            max_dur = self.times[-1]
            self.window_duration = min(max_dur, self.window_duration * 1.25)
            self._update_display()

        
        elif event.key() == Qt.Key_Enter - 1: 
            curr_time = self.annot_manager._to_next_annotation()
            self.current_time = curr_time
            self._update_display()


        elif event.key() == Qt.Key_Delete: 
            self.annot_manager._toggle_annotations()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    from mne.brainheart.load_reference_dataset import load

    raw = load()
    freqs = np.arange(50) + 2

    picks = np.arange(2) + 65

    tf = raw.copy().pick(picks).compute_tfr(method = "morlet", freqs = freqs, decim = 100)

    browser = TFRBrowser(tf,
                         raw = raw, 
                         )
    browser.show()
    sys.exit(app.exec_())

    