from PyQt5.QtWidgets import (
    QWidget, 
    QHBoxLayout, 
    QPushButton, 
    QLabel)


from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt

class ChannelManager(QWidget):
    
    channel_changed = pyqtSignal(int)

    def __init__(
            self, 
            ch_names: list[str], 
            initial_channel: int = 0): 
        super().__init__()

        self.ch_names = ch_names
        self.n_channels = len(ch_names)
        self.current_channel = initial_channel
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

        self.update_channel_label()

    def update_channel_label(self): 
        self.ch_label.setText(
            f"Channel: {self.ch_names[self.current_channel]}({self.current_channel + 1}/{len(self.ch_names)})"
        )
    
    def _prev_channel(self): 
        if self.current_channel > 0: 
            self.current_channel -= 1
            self.update_channel_label()
            self.channel_changed.emit(self.current_channel)
    
    def _next_channel(self): 
        if self.current_channel < self.n_channels - 1: 
            self.current_channel += 1
            self.update_channel_label()
            self.channel_changed.emit(self.current_channel)

    def set_channel(self, channel): 
        self.current_channel = channel
        self.update_channel_label()
