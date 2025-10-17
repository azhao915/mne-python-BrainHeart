from PyQt5.QtGui import QKeyEvent
import mne 
import numpy as np

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import sys

class TFRBrowser(QMainWindow):
    def __init__(self, 
                 tf: mne.time_frequency.BaseTFR, 
                 raw: mne.io.BaseRaw | None = None,
                 dB: float = True,
                 window_duration: float = 10.0, 
                 current_time: float = 0.0,
                 ):
        super().__init__()
        self.data = tf.data # (n_chan, n_freqs, n_times)
        self.dB = dB
        if self.dB: 
            self.data = np.log10(self.data)
        self.times = tf.times
        self.sfreq = tf.sfreq
        self.freqs = tf.freqs

        # If given, then have the raw trace, if not then ignore it
        self.raw = raw

        self.ch_names = tf.ch_names

        # Current State
        self.current_channel = 0
        self.window_duration = 10.0
        self.current_time = 0.0
        
        # Window parameters
        self.window_duration = window_duration  # seconds to show
        self.current_time = current_time  # start time
        
        self.setWindowTitle("TFR Browser")
        self.resize(1200, 800)
        
        self._setup_ui()

        QApplication.processEvents()

        self._update_display()
        
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        channel_layout = QHBoxLayout()

        self.prev_ch_btn = QPushButton("◀ Previous Channel")
        self.prev_ch_btn.clicked.connect(self._prev_channel)
        channel_layout.addWidget(self.prev_ch_btn)

        self.ch_label =QLabel()
        self.ch_label.setAlignment(Qt.AlignCenter)
        channel_layout.addWidget(self.ch_label)

        self.next_ch_btn = QPushButton("Next Channel ▶")
        self.next_ch_btn.clicked.connect(self._next_channel)
        channel_layout.addWidget(self.next_ch_btn)

        main_layout.addLayout(channel_layout)
        
        if self.raw is not None: 
            # Time Trace
            self.plot_time_widget = pg.PlotWidget()
            self.plot_time_widget.setLabel("left", "V")
            self.plot_time_widget.setLabel("bottom", "Time (s)")

            main_layout.addWidget(self.plot_time_widget)

            self.line_item = self.plot_time_widget.plot([], [])
        
        self.plot_tfr_widget = pg.PlotWidget()
        self.plot_tfr_widget.setLabel("left", "Frequency (Hz)")
        self.plot_tfr_widget.setLabel("bottom", "Time (s)")
        main_layout.addWidget(self.plot_tfr_widget)

        self.image_item = pg.ImageItem()
        self.plot_tfr_widget.addItem(self.image_item)

        colormap = pg.colormap.get("inferno")
        self.image_item.setColorMap(colormap)

        self.colorbar = pg.ColorBarItem(
            values=(0, 1),
            colorMap=pg.colormap.get('viridis')
        )
        self.colorbar.setImageItem(self.image_item)

        #self.plot_tfr_widget.addItem(self.colorbar)
        
        #Link the X-axes so they zoom/pan together
        if self.raw is not None: 
            self.plot_time_widget.setXLink(self.plot_tfr_widget)

        self._update_channel_label()

    def _update_channel_label(self): 
        self.ch_label.setText(
            f"Channel: {self.ch_names[self.current_channel]}({self.current_channel + 1}/{len(self.ch_names)})"
        )
    
    def _prev_channel(self): 
        if self.current_channel > 0: 
            self.current_channel -= 1
            self._update_channel_label
            self._update_display()
    
    def _next_channel(self): 
        if self.current_channel < len(self.ch_names) - 1: 
            self.current_channel += 1
            self._update_channel_label
            self._update_display()
    
    def _update_display(self): 

        start_idx = int(self.current_time * self.sfreq)
        end_idx = int((self.current_time + self.window_duration)*self.sfreq)    
        end_idx = min(end_idx, len(self.times))

        time_slice = slice(start_idx, end_idx)
        tfr_data = self.data[self.current_channel, :, time_slice]

        self.image_item.setImage(tfr_data.T, autoLevels = True)

        vmin = np.percentile(tfr_data, 1)
        vmax = np.percentile(tfr_data, 99)

        self.image_item.setLevels([vmin, vmax])

        if self.raw is not None: 
            # Update the upper trace
            times = self.current_time + np.arange(end_idx - start_idx)/self.sfreq
            curr_index_in_raw = mne.pick_channels(raw.ch_names, [tf.ch_names[self.current_channel]])[0]
            raw_trace = raw.get_data(
                picks = curr_index_in_raw, 
                return_times = False, 
                start = int(self.current_time*self.sfreq), 
                stop = int((self.current_time + self.window_duration)*self.sfreq)).flatten()
            self.line_item.setData(times, raw_trace)
            '''
            self.plot_time_widget.setXRange(
                self.current_time, 
                self.current_time + self.window_duration, 
                padding = 0
            )
            '''


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

        self._update_channel_label()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None: 
            return
        if event.key() == Qt.Key_Right: 
            self.current_time += self.window_duration / 4
            self.current_time = min(self.current_time,
                                    self.times[-1] - self.window_duration)
            self._update_display()

        elif event.key() == Qt.Key_Left:
            # Scroll left by 1/4 window
            self.current_time -= self.window_duration / 4
            self.current_time = max(0, self.current_time)
            self._update_display()
            
        elif event.key() == Qt.Key_Up:
            # Previous channel
            self._prev_channel()
            
        elif event.key() == Qt.Key_Down:
            # Next channel
            self._next_channel()
            
        elif event.key() == Qt.Key_Home:
            # Decrease window duration
            self.window_duration = max(1.0, self.window_duration * 0.8)
            self._update_display()
            
        elif event.key() == Qt.Key_End:
            # Increase window duration
            max_dur = self.times[-1]
            self.window_duration = min(max_dur, self.window_duration * 1.25)
            self._update_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    from mne.brainheart.load_reference_dataset import load

    raw = load()
    freqs = np.arange(50) + 1
    picks = np.arange(5) + 63

    tf = raw.copy().pick(picks).compute_tfr(method = "morlet", freqs = freqs)

    browser = TFRBrowser(tf, raw = raw)
    browser.show()
    sys.exit(app.exec_())

    