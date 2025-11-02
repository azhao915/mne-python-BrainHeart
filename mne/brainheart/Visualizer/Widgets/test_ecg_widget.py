"""
test_ecg_widget.py - Simple test application for ECGWidget
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent

from mne.brainheart.Visualizer.Widgets.ECGWidget import ECGWidget
import mne


class ECGTestWindow(QMainWindow):
    """Simple test window for ECGWidget"""
    
    def __init__(self, raw):
        super().__init__()
        
        self.raw = raw
        self.current_time = 0.0
        self.window_duration = 10.0
        
        self.setWindowTitle("ECG Widget Test")
        self.resize(1200, 600)
        
        # Apply dark theme
        self._apply_dark_theme()
        
        # Setup UI
        self._setup_ui()
    
    def _apply_dark_theme(self):
        """Apply Arch-style dark theme"""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d0d0d;
                color: #e0e0e0;
                font-family: 'Monospace', 'Courier New', monospace;
            }
            
            QLabel, QCheckBox {
                color: #a0a0a0;
                font-size: 11px;
            }
            
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #2a2a2a;
                background-color: transparent;
            }
            
            QCheckBox::indicator:checked {
                background-color: #88c0d0;
                border: 1px solid #88c0d0;
            }
            
            QCheckBox::indicator:hover {
                border: 1px solid #3a3a3a;
            }
        """)
    
    def _setup_ui(self):
        """Setup the UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(main_layout)
        
        # Create ECG widget
        try:
            self.ecg_widget = ECGWidget(
                raw=self.raw,
                curr_time=self.current_time,
                window_duration=self.window_duration,
                show_peaks=True,
                show_artifacts=True
            )
            main_layout.addWidget(self.ecg_widget)
        except ValueError as e:
            print(f"Error creating ECG widget: {e}")
            return
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard navigation"""
        if event.key() == Qt.Key_Right:
            # Scroll forward
            self.current_time += self.window_duration / 4
            max_time = self.raw.times[-1] - self.window_duration
            self.current_time = min(self.current_time, max_time)
            self.ecg_widget.update_display(curr_time=self.current_time)
            
        elif event.key() == Qt.Key_Left:
            # Scroll backward
            self.current_time -= self.window_duration / 4
            self.current_time = max(0, self.current_time)
            self.ecg_widget.update_display(curr_time=self.current_time)
            
        elif event.key() == Qt.Key_Home:
            # Zoom in (decrease window duration)
            self.window_duration = max(1.0, self.window_duration * 0.8)
            self.ecg_widget.update_display(window_duration=self.window_duration)
            
        elif event.key() == Qt.Key_End:
            # Zoom out (increase window duration)
            max_dur = self.raw.times[-1]
            self.window_duration = min(max_dur, self.window_duration * 1.25)
            self.ecg_widget.update_display(window_duration=self.window_duration)
        
        elif event.key() == Qt.Key_Escape:
            # Close window
            self.close()


def create_synthetic_ecg_raw():
    """
    Create synthetic ECG data for testing when you don't have real data.
    """
    # Create synthetic ECG data
    sfreq = 250  # Hz
    duration = 60  # seconds
    n_samples = int(sfreq * duration)
    times = np.arange(n_samples) / sfreq
    
    # Simulate ECG signal (simplified)
    heart_rate = 70  # bpm
    rr_interval = 60 / heart_rate  # seconds
    
    # Create raw ECG with QRS complexes
    ecg_raw = np.zeros(n_samples)
    r_peak_indices = []
    
    current_time = 1.0  # Start after 1 second
    while current_time < duration - 1:
        peak_idx = int(current_time * sfreq)
        
        # Add QRS complex (simplified)
        qrs_width = int(0.1 * sfreq)  # 100ms
        for i in range(-qrs_width//2, qrs_width//2):
            if 0 <= peak_idx + i < n_samples:
                t = i / sfreq
                ecg_raw[peak_idx + i] += np.exp(-t**2 / 0.001) * (1 if i < 0 else -0.3)
        
        r_peak_indices.append(peak_idx)
        current_time += rr_interval + np.random.randn() * 0.05  # Add variability
    
    # Add noise
    noise = np.random.randn(n_samples) * 0.05
    ecg_raw += noise
    
    # Create clean version (smoothed)
    from scipy.ndimage import gaussian_filter1d
    ecg_clean = gaussian_filter1d(ecg_raw, sigma=2)
    
    # Create R peaks channel (binary)
    ecg_r_peaks = np.zeros(n_samples)
    ecg_r_peaks[r_peak_indices] = 1
    
    # Calculate heart rate
    ecg_rate = np.zeros(n_samples)
    for i in range(len(r_peak_indices) - 1):
        start_idx = r_peak_indices[i]
        end_idx = r_peak_indices[i + 1]
        interval = (end_idx - start_idx) / sfreq
        rate = 60 / interval  # bpm
        ecg_rate[start_idx:end_idx] = rate
    
    ecg_rate = ecg_rate / 70
    
    # Create quality metric (random for demo)
    ecg_quality = 0.8 + 0.2 * np.random.rand(n_samples)
    
    # Create MNE Raw object
    ch_names = ['ECG_Raw', 'ECG_Clean', 'ECG_R_Peaks', 'ECG_Rate', 'ECG_Quality']
    ch_types = ['ecg', 'ecg', 'ecg', 'ecg', 'ecg']
    
    data = np.vstack([ecg_raw, ecg_clean, ecg_r_peaks, ecg_rate, ecg_quality])
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    
    return raw


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Option 1: Load your real data
    # Uncomment this if you have your load function:
    # from mne.brainheart.load_reference_dataset import load
    # raw = load()

    from mne.brainheart.load_reference_dataset import load
    from mne.brainheart.ecg_wrappers import ecg_process_neurokit
    from mne.brainheart.loading.ecg_loading import annotate_valid_ecg_periods, identify_ecg_channel
    # Option 2: Use synthetic data for testing
    print("Creating synthetic ECG data...")
    raw = load(0)
    ecg_process_neurokit(raw, "EKG")
    print(f"Created {len(raw.ch_names)} ECG channels: {raw.ch_names}")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    print("\nControls:")
    print("  Left/Right arrows: Scroll through data")
    print("  Home/End: Zoom in/out")
    print("  Checkboxes: Toggle peaks and artifacts")
    print("  ESC: Close window")
    
    # Create and show window
    window = ECGTestWindow(raw)
    window.show()
    
    sys.exit(app.exec_())