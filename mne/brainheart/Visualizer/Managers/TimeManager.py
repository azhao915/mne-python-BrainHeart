from PyQt5.QtCore import QObject, pyqtSignal
from mne.brainheart.Visualizer.Managers.AnnotationsManager import AnnotationsManager

class TimeManager(QObject): 
    
    # Signals
    time_params_changed = pyqtSignal(float, float)

    def __init__(
            self, 
            initial_time: float = 0.0, 
            initial_duration: float = 10.0, 
            max_time: float | None = None
    ): 
        super().__init__()
        self.current_time = initial_time
        self.window_duration = initial_duration
        self.max_time = max_time

        # Constraints
        self.min_time = 0.0
        self.min_duration = 0.5

        self.widgets = []

        self.annot_manager = None

    def register_widget(self, widget): 
        if not hasattr(widget, "_update_display"): 
            raise ValueError("widget must have an _update_display method")
        self.widgets.append(widget)

        self.time_params_changed.connect(
            lambda t, d: widget._update_display(
                curr_time = t, 
                window_duration = d
            )
        )

        self.widgets.append(widget)

        # TEMPORARY FIX, need to fix later by having a more unified representation

        if hasattr(widget, "setXLink"): 
            curr_plot_widget = widget
        else: 
            curr_plot_widget = widget.plot_tfr_widget
        
        if hasattr(self.widgets[0], "setXLink"): 
            first_plot_widget = self.widgets[0]
        else:
            first_plot_widget = self.widgets[0].plot_tfr_widget

        curr_plot_widget.setXLink(first_plot_widget)

        # Initial Update
        widget._update_display(
            curr_time = self.current_time, 
            window_duration = self.window_duration
        )

        # Now link the X-Axes
        if not len(self.widgets): 
            self.widgets[0].setXLink(widget)
    
    def unregister_widget(self, widget): 
        # Incomplete, and might never be used
        if not widget in self.register_widget: 
            return
        self.time_params_changed.disconnect()
        self.widgets.remove(widget)

        # Now reconnect the old widgets
        for w in self.widgets: 
            self.time_params_changed.connect(
                lambda t, d: widget._update_display(
                    curr_time = t, 
                    window_duration = d
                )
            )

    def register_annotations_manager(self, manager: AnnotationsManager): 
        if not hasattr(manager, "update_annotations"): 
            raise ValueError("manager must have a update_annotations method")
        if not hasattr(manager, "_to_next_annotation"): 
            raise ValueError("manager must have a _to_next_annotation method")
        self.annot_manager = manager
        self.time_params_changed.connect(
            lambda t, d: manager.update_annotations(
                current_time = t, 
                window_duration = d
            )
        )
        # first update
        manager.update_annotations(
            current_time = self.current_time, 
            window_duration = self.window_duration
        )
    
    def to_next_annotation(self): 
        if self.annot_manager is None or not self.annot_manager.display_annotations: 
            return
        current_time = self.annot_manager._to_next_annotation()
        if current_time is not None: 
            self.set_time(current_time)
        

    def set_time(self, new_time: float): 
        if new_time is None: 
            return
        new_time = max(self.min_time, new_time)
        if self.max_time is not None: 
            new_time = min(new_time, self.max_time)
        if new_time != self.current_time: 
            self.current_time = new_time
            self.time_params_changed.emit(
                self.current_time, 
                self.window_duration
            )
    
    def set_window_duration(self, new_duration: float): 
        if new_duration is None: 
            return
        new_duration = max(self.min_duration, new_duration)
        if self.max_time is not None: 
            new_duration = min(self.max_time, new_duration)
        
        if new_duration != self.window_duration: 
            self.window_duration = new_duration

            if self.max_time is not None: 
                max_valid_time = self.max_time - self.window_duration
                if self.current_time > max_valid_time: 
                    self.current_time = self.set_time(max_valid_time)
                    
            self.time_params_changed.emit(self.current_time, self.window_duration)

    def scroll_forward(self, prop = 0.25): 
        self.set_time(self.current_time + self.window_duration * prop)
    
    def scroll_backward(self, prop = 0.25): 
        self.set_time(self.current_time - self.window_duration * prop)

    def zoom_in(self, prop = 0.8): 
        self.set_window_duration(self.window_duration * prop)
    
    def zoom_out(self, prop = 1.25): 
        self.set_window_duration(self.window_duration * prop)


