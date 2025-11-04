from PyQt5.QtCore import QObject, pyqtSignal


class TimeManager(QObject): 
    
    # Signals
    time_changed = pyqtSignal(float)
    window_duration_changed = pyqtSignal(float)
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

        # Initial Update
        widget._update_display(
            curr_time = self.current_time, 
            window_duration = self.window_duration
        )
    
    def unregister_widget(self, widget): 
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
        

    def set_time(self, new_time: float): 
        if new_time is None: 
            return
        new_time = max(self.min_time, new_time)
        if self.max_time is not None: 
            new_time = min(new_time, self.max_time)
        if new_time != self.current_time: 
            self.current_time = new_time
            self.time_changed.emit(self.current_time)
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

            time_changed = False
            if self.max_time is not None: 
                max_valid_time = self.max_time - self.window_duration
                if self.current_time > max_valid_time: 
                    self.current_time = self.set_time(max_valid_time)
                    time_changed = True
                
            self.window_duration_changed.emit(self.window_duration)
            if time_changed: 
                self.time_params_changed.emit(self.current_time, self.window_duration)

    def scroll_forward(self, prop = 0.25): 
        self.set_time(self.current_time + self.window_duration * prop)
    
    def scroll_backward(self, prop = 0.25): 
        self.set_time(self.current_time - self.window_duration * prop)

    def zoom_in(self, prop = 0.8): 
        self.set_window_duration(self.window_duration * prop)
    
    def zoom_out(self, prop = 1.25): 
        self.set_window_duration(self.window_duration * prop)


