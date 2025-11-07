from PyQt5.QtWidgets import (
    QWidget, 
    QHBoxLayout, 
    QPushButton, 
    QLabel)


from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt

class ChannelManager(QWidget):
    
    channel_name_changed = pyqtSignal(str)

    def __init__(
            self, 
            ch_names: list[str], 
            initial_channel: int = 0): 
        super().__init__()

        self.ch_names = ch_names
        self.n_channels = len(ch_names)
        self.current_channel = initial_channel
        self.curr_chan_name = self.ch_names[self.current_channel]

        self.widgets = []

        self._setup_ui()
    
    def _setup_ui(self):

        layout = QHBoxLayout()

        self.prev_ch_btn = QPushButton("◀ Previous Channel")
        self.prev_ch_btn.clicked.connect(self._prev_channel)
        layout.addWidget(self.prev_ch_btn)

        self.ch_label =QLabel()
        self.ch_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.ch_label)

        self.next_ch_btn = QPushButton("Next Channel ▶")
        self.next_ch_btn.clicked.connect(self._next_channel)
        layout.addWidget(self.next_ch_btn)

        self.setLayout(layout)

        self.redraw_channel_label()

    
    def register_widget(self, widget): 
        if not hasattr(widget, "_update_display"): 
            raise ValueError("widget must have an _update_display method")
        self.channel_name_changed.connect(
            lambda chan_name: widget._update_display(
                curr_channel_name = chan_name
            )
        )
        # First Update
        widget._update_display(curr_channel_name = self.curr_chan_name)


    def redraw_channel_label(self): 
        self.ch_label.setText(
            f"Channel: {self.curr_chan_name}({self.current_channel + 1}/{len(self.ch_names)})"
        )
    
    def _prev_channel(self): 
        if self.current_channel > 0: 
            self.set_channel(self.current_channel - 1)
    
    def _next_channel(self): 
        if self.current_channel < self.n_channels - 1: 
            self.set_channel(self.current_channel + 1)

    def set_channel(self, channel): 
        self.current_channel = channel
        self.curr_chan_name = self.ch_names[self.current_channel]
        self.redraw_channel_label()
        self.channel_name_changed.emit(self.curr_chan_name)

    def update_channel_name(self, chan_name): 
        if chan_name not in self.ch_names: 
            return
        chan_index = self.ch_names.index(chan_name)
        self.set_channel(chan_index)
