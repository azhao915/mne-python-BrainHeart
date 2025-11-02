from PyQt5.QtCore import QObject, pyqtSignal
import pyqtgraph as pf
from PyQt5.QtCore import Qt

class AnnotationsManager(QObject): 
    
    def __init__(self, raw): 
        self.raw = raw
        self.plots = {}
        self.display_annotations = True

        self.line_params = dict(            
            angle = 90, 
            pen = pg.mkPen(color = "b", width = 2, style = Qt.DashLine), 
            movable = False, 
            label = desc, 
            labelOpts = {"position": 0.95, "color": "b"}
        )

        self.region_params = dict(
                brush = pg.mkBrush(0, 0, 255, 60), 
                movable = False
            )
    
    def register_plot(self, plot_widget, plot_name: str | None = None): 
        if plot_name is None: 
            plot_name = str(plot_widget)
        
        self.plots[plot_name] = {
            "widget": plot_widget, 
            "lines": [], 
            "regions": []
        }

    def unregister_plot(self, plot_name):
        if plot_name in self.plots:
            self.clear_annotations(plot_name)
            del self.plots[plot_name]
    
    def update_annotations(self, start_time, end_time): 
        # Get all the annotations
        annnotations = self.raw.annotations
        onsets, durations, descs = annotations.onset, annotations.duration, annotations.description
        ends = onsets + durations
        # Get all the annotations visible
        mask_onset_valid = onsets <= end_time
        mask_end_valid = ends >= start_time
        valid_mask = np.logical_and(mask_onset_valid, mask_end_valid) 
        if not np.any(valid_mask): 
            return
        onsets, durations, descs = onsets[valid_mask], durations[valid_mask], descs[valid_mask]
        # Trim what is necessary
        onsets[onsets <= start_time] = start_time
        ends[ends >= end_time] = end_time
        # Plot these 
        for onset, end, description in zip(onsets, ends, descs):
            duration = end - onset
            for plot_name, plot_data in self.plot.items(): 
                self._add_annotations_to_plot(
                    plot_data, onset, duration, description
                )

    def _add_annotation_to_plot(
            self, 
            plot_data, 
            onset, 
            duration, 
            description
    ):
        if self.raw.annotations is None and not self.display_annotations: 
            return
        line = pg.InfiniteLine(
            pos = onset,
            **self.line_params
        )
        plot_data["widget"].addItem(line)
        plot_data["lines"].append(line)

        if end > onset: 
            region = pg.LinearRegionItem(
                values = [onset, end], 
                **self.region_params
            )
            plot_data["widget"].addItem(region)
            plot_data["regions"].append(region)
    
    def _clear_annotations(self, plot_name = None):
        available_plot_names = list(self.plots.keys())
        if plot_name is None: 
            plots_to_clear = available_plot_names
        else: 
            plots_to_clear = [plot for plot in plot_name if plot in available_plot_names]
        
        for name in plots_to_clear: 
            plot_data = self.plots[name]
            for line in plot_data["lines"]: 
                plot_data["widget"].removeItem(line)
            plot_data["lines"] = []
            for region in plot_data["regions"]: 
                plot_data["widget"].removeItem(region)
            plot_data["regions"] = []
    