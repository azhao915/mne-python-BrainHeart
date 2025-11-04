from PyQt5.QtCore import QObject, pyqtSignal
import pyqtgraph as pf
from PyQt5.QtCore import Qt

import pyqtgraph as pg

import numpy as np

class AnnotationsManager(QObject): 
    
    def __init__(
            self, 
            annotations, 
            current_time, 
            window_duration):
        super().__init__()
        self.annotations = annotations
        self.plots = {}
        self.display_annotations = True

        self.current_time = current_time
        self.window_duration = window_duration

        self.line_params = dict(            
            angle = 90, 
            pen = pg.mkPen(color = "b", width = 2, style = Qt.DashLine), 
            movable = False, 
            labelOpts = {"position": 0.95, "color": "b"}
        )

        self.region_params = dict(
                brush = pg.mkBrush(0, 0, 255, 60), 
                movable = False
            )

    
    def register_plot(
            self, 
            plot_name: str, 
            plot_widget): 

        if plot_name is None: 
            plot_name = str(plot_widget)
        
        self.plots[plot_name] = {
            "widget": plot_widget, 
            "lines": [], 
            "regions": []
        }

    def unregister_plot(self, plot_name):
        if plot_name in self.plots:
            self._clear_annotations(plot_name)
            del self.plots[plot_name]
    

    def toggle_annotations(self): 
        self.display_annotations = not self.display_annotations
        self.redraw_annotations()

    
    def _to_next_annotation(self): 
        if self.annotations is None or not self.display_annotations: 
            return
        onsets = self.annotations.onset
        # Assume already sorted
        curr_mid_window_time = self.current_time + self.window_duration/2
        next_annotation_indices = np.where(onsets > curr_mid_window_time)[0]
        if not len(next_annotation_indices): 
            return
        next_annotation_index = next_annotation_indices[0]
        '''
        self.current_time = onsets[next_annotation_index] - 0.5*self.window_duration
        # Have the annotation onset be at the middle of the display screen
        self.current_time = np.max(self.current_time, 0)
        self.redraw_annotations()
        '''
        # Temporary Fix
        current_time = onsets[next_annotation_index] - 0.5*self.window_duration
        current_time = np.max(current_time, 0)
        return current_time

    def update_annotations(
            self, 
            current_time: float | None = None, 
            window_duration: float | None = None): 
        if not current_time is None: 
            self.current_time = current_time
        if not window_duration is None: 
            self.window_duration = window_duration

        self.redraw_annotations()

    def redraw_annotations(
            self
    ):
        if self.annotations is None: 
            return
        self._clear_annotations()
        if not self.display_annotations: 
            return
        
        end_time = self.current_time + self.window_duration

        # Get all the annotations
        annotations = self.annotations
        onsets, durations, descs = annotations.onset, annotations.duration, annotations.description
        ends = onsets + durations
        # Get all the annotations visible
        mask_onset_valid = onsets <= end_time
        mask_end_valid = ends >= self.current_time
        valid_mask = np.logical_and(mask_onset_valid, mask_end_valid) 
        if not np.any(valid_mask): 
            return
        onsets, durations, descs = onsets[valid_mask], durations[valid_mask], descs[valid_mask]
        # Trim what is necessary
        onsets[onsets <= self.current_time] = self.current_time
        ends[ends >= end_time] = end_time
        # Plot these 
        for onset, end, description in zip(onsets, ends, descs):
            duration = end - onset
            for plot_name, plot_data in self.plots.items(): 
                self._add_annotation_to_plot(
                    plot_data, onset, duration, description
                )

    def _add_annotation_to_plot(
            self, 
            plot_data, 
            onset, 
            duration, 
            description
    ):
        if self.annotations is None and not self.display_annotations: 
            return
        line = pg.InfiniteLine(
            pos = onset,
            label = description,
            **self.line_params
        )
        plot_data["widget"].addItem(line)
        plot_data["lines"].append(line)

        if duration > onset: 
            region = pg.LinearRegionItem(
                values = [onset, onset + duration], 
                **self.region_params
            )
            plot_data["widget"].addItem(region)
            plot_data["regions"].append(region)
    
    def _clear_annotations(self, plot_name = None):
        available_plot_names = list(self.plots.keys())
        if plot_name is None: 
            plots_to_clear = available_plot_names
        else: 
            plots_to_clear = [plot_name] if plot_name in available_plot_names else []
        
        for name in plots_to_clear: 
            plot_data = self.plots[name]
            for line in plot_data["lines"]: 
                plot_data["widget"].removeItem(line)
            plot_data["lines"] = []
            for region in plot_data["regions"]: 
                plot_data["widget"].removeItem(region)
            plot_data["regions"] = []
    