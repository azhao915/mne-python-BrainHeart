import pyqtgraph as pg
import numpy as np

class TimeWidget(pg.PlotWidget): 
    def __init__(
            self, 
            raw, 
            curr_channel, 
            curr_time, 
            window_duration): 
        super().__init__()

        self.data = raw.get_data(
            return_times = False
        )
        self.ch_names = raw.ch_names
        self.sfreq = raw.info["sfreq"]

        self.curr_channel = curr_channel

        # Window Parameters
        self.curr_time = curr_time
        self.window_duration = window_duration

        self._setup_ui()
    
    def _setup_ui(self):
        self.setLabel("left", "V")
        self.setLabel("bottom", "Time (s)")

        self.line_item = self.plot([], [])
    
    def _update_display(
            self, 
            curr_channel_name: str | None = None,
            curr_channel: int | None = None,
            curr_time: float | None = None, 
            window_duration: float | None = None,
    ): 
        if curr_channel_name is not None: 
            if curr_channel_name in self.ch_names: 
                curr_channel = self.ch_names.index(curr_channel_name)
            else: 
                curr_channel = None
        if curr_channel is not None: 
            self.curr_channel = curr_channel
        if curr_time is not None: 
            self.curr_time = curr_time 
        if window_duration is not None: 
            self.window_duration = window_duration
        self.redraw()
    
    def redraw(
            self
    ): 
        start_idx_raw = int(self.curr_time*self.sfreq)
        end_idx_raw = int((self.curr_time + self.window_duration)*self.sfreq)
        times = self.curr_time + np.arange(end_idx_raw - start_idx_raw)/self.sfreq
        raw_trace = self.data[self.curr_channel, start_idx_raw:end_idx_raw].flatten()
        self.line_item.setData(times, raw_trace)
        
