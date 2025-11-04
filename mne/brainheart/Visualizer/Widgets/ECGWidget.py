import pyqtgraph as pg
import numpy as np

from mne.brainheart.ecg_channel_names_enum import ECG_Channels
from mne.brainheart.ecg_annotations_enum import ECG_Annotations

from mne import channel_indices_by_type

from pens import Colors, get_ecg_style, is_peak_channel, is_continuous_channel

from PyQt5.QtWidgets import (
    QVBoxLayout, 
    QWidget, 
    QCheckBox, 
    QHBoxLayout
)

from mne.brainheart.loading.ecg_loading import _select_single_ecg_channel

class ECGWidget(QWidget): 
    def __init__(
            self, 
            raw, 
            curr_time = 0.0, 
            window_duration = 10.0, 
            show_peaks = True,
            show_artifacts = True,
    ): 
        super().__init__()
        ecg_ch_indices_in_raw = channel_indices_by_type(raw.info)["ecg"]
        if not len(ecg_ch_indices_in_raw): 
            raise ValueError("There are no ECG Channels")
        
        self.raw = raw
        self.sfreq = raw.info["sfreq"]

        self.curr_time = curr_time
        self.window_duration = window_duration
        self.show_peaks = show_peaks
        self.show_artifacts = show_artifacts

        raw_data = raw.get_data(return_times = False)

        self.left_axis_channels = {}
        self.right_axis_channels = {}
        self.peak_channels = {}

        for chan in ECG_Channels: 
            chan_name = chan.value
            if "Phase" in chan_name: 
                continue
            if "Raw" in chan_name: 
                continue
            if not chan_name in raw.ch_names: 
                continue
            chan_index = raw.ch_names.index(chan_name)
            data = raw_data[chan_index, :].flatten()
            style = get_ecg_style(chan_name)
            
            chan_info = {
                "data": data, 
                "style": style, 
                "line_item": None, 
                "scatter_item": None, 
                "visible": True
            }

            if is_peak_channel(chan_name) or "Onset" in chan_name or "Offset" in chan_name:
                self.peak_channels[chan_name] = chan_info
            elif self.is_in_left_axis(chan_name): 
                self.left_axis_channels[chan_name] = chan_info
            else: 
                self.right_axis_channels[chan_name] = chan_info

        if not (self.left_axis_channels or self.right_axis_channels or self.peak_channels): 
            raise ValueError("No matching ECG channels found")
        
        # Data to use for y-value of the peaks
        ecg_index = _select_single_ecg_channel(raw)
        self.y_value_peaks = raw_data[ecg_index, :].flatten()

        self._setup_ui()
        self.redraw()

    def is_in_left_axis(self, chan_name): 
        return not (
            "Rate" in chan_name or 
            "Quality" in chan_name or 
            "Phase" in chan_name)        

    def _setup_ui(self): 
        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(
            0, 0, 0, 0
        )
        self.setLayout(main_layout)

        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(5)

        self._setup_checkboxes(controls_layout)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        self.plot_widget = pg.PlotWidget()
        self._style_plot()
        main_layout.addWidget(self.plot_widget)

        self.right_viewbox = pg.ViewBox()
        self.plot_widget.scene().addItem(self.right_viewbox)

        # Create Right Axis
        self.right_axis = pg.AxisItem("right")
        self.plot_widget.plotItem.layout.addItem(
            self.right_axis, 2, 3
        )
        self.right_axis.linkToView(self.right_viewbox)
        self.right_axis.setLabel("HeartRate (bpm)")

        # Style the Right Axis
        self.right_axis.setPen(color = Colors.BORDER, width = 1)
        self.right_axis.setTextPen(Colors.TEXT_SECONDARY)

        # Link X axes and handle view resizing
        self.right_viewbox.setXLink(
            self.plot_widget.plotItem.vb
        )
        self._update_views()
        self.plot_widget.plotItem.vb.sigResized.connect(
            self._update_views
        )

        for chan_name, chan_data in self.left_axis_channels.items(): 
            style = chan_data["style"]
            chan_data["line_item"] = self.plot_widget.plot(
                [], [], 
                pen = style["pen"], 
                name = chan_name
            ) 

        for chan_name, chan_data in self.right_axis_channels.items(): 
            style = chan_data["style"]
            line_item = pg.PlotCurveItem(
                [], [], 
                pen = style["pen"], 
            ) 
            self.right_viewbox.addItem(line_item)
            chan_data["line_item"] = line_item
        
        for chan_name, chan_data in self.peak_channels.items(): 
            style = chan_data["style"]
            scatter_item = pg.ScatterPlotItem(
                size=style.get('size', 8),
                pen=style.get('pen'),
                brush=style.get('brush'),
                symbol=style.get('symbol', 'o'),
                name=chan_name
            )
            self.plot_widget.addItem(scatter_item)
            chan_data["scatter_item"] = scatter_item
        
        self.plot_widget.addLegend(offset = (10, 10))


    def _update_views(self): 
        self.right_viewbox.setGeometry(
            self.plot_widget.plotItem.vb.sceneBoundingRect()
        )
        self.right_viewbox.linkedViewChanged(
            self.plot_widget.plotItem.vb, self.right_viewbox.XAxis
        )


    def _setup_checkboxes(self, controls_layout): 
        self.show_peaks_cb = QCheckBox("Show Peaks")
        self.show_peaks_cb.setChecked(self.show_peaks)
        self.show_peaks_cb.stateChanged.connect(self._on_show_peaks_changed)
        controls_layout.addWidget(self.show_peaks_cb)

        self.show_artifacts_cb = QCheckBox("Show Artifacts")
        self.show_artifacts_cb.setChecked(self.show_artifacts)
        self.show_artifacts_cb.stateChanged.connect(self._on_show_artifacts_changed)
        controls_layout.addWidget(self.show_artifacts_cb)

    def _style_plot(self):
        self.plot_widget.setBackground(Colors.MAIN_BG)
        self.plot_widget.getAxis('left').setPen(
            color=Colors.BORDER, 
            width=1)
        self.plot_widget.getAxis('bottom').setPen(
            color=Colors.BORDER, 
            width=1)
        self.plot_widget.getAxis('left').setTextPen(Colors.TEXT_SECONDARY)
        self.plot_widget.getAxis('bottom').setTextPen(Colors.TEXT_SECONDARY)
        
        self.plot_widget.showGrid(
            x=True, 
            y=True, 
            alpha=0.1)
        
        self.plot_widget.setLabel(
            "left", 
            "ECG", 
            color=Colors.TEXT_SECONDARY, 
            **{'font-size': '10pt'})
        self.plot_widget.setLabel(
            "bottom", 
            "Time (s)", 
            color=Colors.TEXT_SECONDARY, 
            **{'font-size': '10pt'})

    def _on_show_peaks_changed(self, show_peaks): 
        self.show_peaks = show_peaks
        self.redraw()

    def _on_show_artifacts_changed(self, show_artifacts): 
        self.show_artifacts = show_artifacts
        self.redraw()

    def update_display(
            self, 
            curr_time: float | None = None, 
            window_duration: float | None = None
    ): 
        if curr_time is not None: 
            self.curr_time = curr_time
        if window_duration is not None: 
            self.window_duration = window_duration

        self.redraw()

    def redraw(self): 
        start_idx = int(self.curr_time * self.sfreq)
        end_idx = int((self.curr_time + self.window_duration)*self.sfreq)

        n_samples = end_idx - start_idx
        times = self.curr_time + np.arange(
            n_samples
        )/self.sfreq

        # Update the continuous Channels
        for chan_name, chan_data in self.left_axis_channels.items():
            trace = chan_data["data"][start_idx:end_idx]
            line_item = chan_data["line_item"]
            if line_item is not None: 
                line_item.setData(times, trace)
        
        for chan_name, chan_data in self.right_axis_channels.items():
            trace = chan_data["data"][start_idx:end_idx]
            line_item = chan_data["line_item"]
            if line_item is not None: 
                line_item.setData(times, trace)
        
        # Update the Peaks
        for chan_name, chan_data in self.peak_channels.items(): 
            scatter_item = chan_data["scatter_item"]

            is_artifact = "fixpeaks" in chan_name
            if (is_artifact and not self.show_artifacts) or (not is_artifact and not self.show_peaks): 
                scatter_item.setData([], [])
                continue
            
            peak_data = chan_data["data"][start_idx:end_idx]
            peak_indices = np.where(peak_data)[0]
            if len(peak_indices): 
                peak_times = times[peak_indices]
                peak_values = self.y_value_peaks[start_idx + peak_indices]
                scatter_item.setData(peak_times, peak_values)
            else: 
                scatter_item.setData([], [])
        
        self.plot_widget.setXRange(
            self.curr_time, 
            self.curr_time + self.window_duration, 
            padding = 0
        )

        self.plot_widget.enableAutoRange(axis = "y")
        self.right_viewbox.enableAutoRange(axis = pg.ViewBox.YAxis)

    def get_plot_widget(self): 
        return self.plot_widget

