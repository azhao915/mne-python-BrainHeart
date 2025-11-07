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

from mne.brainheart.Visualizer.Managers.ChannelManager import ChannelManager
from mne.brainheart.Visualizer.Managers.AnnotationsManager import AnnotationsManager
from mne.brainheart.Visualizer.Managers.TimeManager import TimeManager
from mne.brainheart.Visualizer.Managers.KeypressManager import KeybindingManager

from mne.brainheart.Visualizer.Widgets.TimeWidget import TimeWidget
from mne.brainheart.Visualizer.Widgets.TFRWidget import TFRWidget

from mne.brainheart.testing.test_spectrum import welch_with_CI

from mne.brainheart.Visualizer.StyleSheets import minimalist_sheet

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

        self.setStyleSheet(minimalist_sheet)

        self.data = tf.data # (n_chan, n_freqs, n_times)
        self.dB = dB
        if self.dB: 
            self.data = 20*np.log10(self.data)
        self.times = tf.times
        self.sfreq_tf = tf.sfreq
        self.freqs = tf.freqs

        # If given, then have the raw trace, if not then ignore it
        self.raw = raw

        self.tf = tf

        # Initialize the annotations if possible
        if annotations is None and raw is not None: 
            annotations = raw.annotations
        self.annotations = annotations
        self.display_annotations = True

        self.ch_names = tf.ch_names
        self.n_channels = len(self.ch_names)

        # Current State
        self.current_channel = 0
        # Window parameters
        self.window_duration = window_duration  # seconds to show
        self.current_time = current_time  # start time

        # Annotations Manager
        self.annot_manager = AnnotationsManager(
            annotations = self.annotations,
            current_time = self.current_time, 
            window_duration = self.window_duration 
        )

        # Time Manager
        # Might move this if add a Widget component
        self.time_manager = TimeManager(
            initial_time = self.current_time,
            initial_duration = self.window_duration, 
            max_time = self.times[-1]
        )

        self.time_manager.register_annotations_manager(
            self.annot_manager
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
        
        main_layout.addWidget(self.channel_manager)

        # Have to add the Key Press Events here
        self.keybinding_manager = KeybindingManager(self)
        
        if self.raw is not None: 
            # Time Trace
            chan_index_in_raw = self.raw.ch_names.index(self.ch_names[self.current_channel])
            self.plot_time_widget = TimeWidget(
                raw = self.raw, 
                curr_channel = chan_index_in_raw, 
                curr_time = self.current_time, 
                window_duration = self.window_duration
            )

            self.annot_manager.register_plot(
                plot_name = "time", 
                plot_widget = self.plot_time_widget)
            
            self.time_manager.register_widget(
                self.plot_time_widget
            )

            self.channel_manager.register_widget(self.plot_time_widget)
            
            main_layout.addWidget(self.plot_time_widget)
        
        self.tfr_widget = TFRWidget(
            tf = self.tf, 
            curr_channel = self.current_channel, 
            curr_time = self.current_time, 
            window_duration = self.window_duration, 
            dB = self.dB
        )
        main_layout.addWidget(self.tfr_widget)

        self.time_manager.register_widget(
            self.tfr_widget
        )

        self.annot_manager.register_plot(
            plot_name = "tfr", 
            plot_widget = self.tfr_widget.plot_tfr_widget
        )

        self.channel_manager.register_widget(
            self.tfr_widget
        )


    def _on_channel_changed(self, new_channel): 
        self.current_channel = new_channel
        self._update_display()

    def _update_display(self): 

        self.annot_manager.update_annotations(
            self.current_time, 
            self.window_duration)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None: 
            return
        if not self.keybinding_manager.handle_key_press(event.key()): 
            super().keyPressEvent(event)


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

    