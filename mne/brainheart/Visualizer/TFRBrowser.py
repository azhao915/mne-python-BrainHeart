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

from TimePlotting import TimePlotting
from TFRPlotting import TFRPlotting

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

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d0d0d;
                color: #e0e0e0;
                font-family: 'Monospace', 'Courier New', monospace;
            }
            
            QLabel {
                color: #a0a0a0;
                font-size: 11px;
                font-weight: normal;
                padding: 2px;
            }
            
            QPushButton {
                background-color: transparent;
                color: #e0e0e0;
                border: 1px solid #2a2a2a;
                padding: 4px 10px;
                border-radius: 0px;
                font-size: 10px;
                font-family: 'Monospace', 'Courier New', monospace;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            QPushButton:hover {
                background-color: #1a1a1a;
                border: 1px solid #3a3a3a;
                color: #ffffff;
            }
            
            QPushButton:pressed {
                background-color: #0a0a0a;
                border: 1px solid #1a1a1a;
            }
            
            QPushButton:disabled {
                background-color: transparent;
                color: #404040;
                border: 1px solid #1a1a1a;
            }
            
            /* Minimal separator lines */
            QFrame {
                border: none;
                background-color: #1a1a1a;
            }
        """)

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
            self.plot_time_widget = TimePlotting(
                raw = self.raw, 
                curr_channel = self.current_channel, 
                curr_time = self.current_time, 
                window_duration = self.window_duration
            )

            self.annot_manager.register_plot(
                plot_name = "time", 
                plot_widget = self.plot_time_widget)
            
            main_layout.addWidget(self.plot_time_widget)
        
        self.tfr_widget = TFRPlotting(
            tf = tf, 
            curr_channel = self.current_channel, 
            curr_time = self.current_time, 
            window_duration = self.window_duration, 
            dB = self.dB
        )
        main_layout.addWidget(self.tfr_widget)

        self.annot_manager.register_plot(
            plot_name = "tfr", 
            plot_widget = self.tfr_widget.plot_tfr_widget
        )

        #Link the X-axes so they zoom/pan together
        if self.raw is not None: 
            self.plot_time_widget.setXLink(self.tfr_widget.plot_tfr_widget)


    def _on_channel_changed(self, new_channel): 
        self.current_channel = new_channel
        self._update_display()

    
    def _update_display(self): 
        self.tfr_widget._update_display(
            curr_channel = self.current_channel, 
            curr_time = self.current_time, 
            window_duration = self.window_duration
        )

        if self.raw is not None: 
            # Update the Upper Trace
            self.plot_time_widget._update_display(
                curr_channel_name = tf.ch_names[self.current_channel], 
                curr_time = self.current_time, 
                window_duration = self.window_duration
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

    