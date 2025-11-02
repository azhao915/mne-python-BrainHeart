

minimalist_sheet = """
QMainWindow, QWidget {
        background-color: #0d0d0d;
        color: #e0e0e0;
        font-family: 'Monospace', 'Courier New', monospace;
    }
    
    QLabel {
        color: #a0a0a0;
        font-size: 11px;
        font-weight: normal;
        padding: 2px;
    }
    
    QPushButton {
        background-color: transparent;
        color: #e0e0e0;
        border: 1px solid #2a2a2a;
        padding: 4px 10px;
        border-radius: 0px;
        font-size: 10px;
        font-family: 'Monospace', 'Courier New', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    QPushButton:hover {
        background-color: #1a1a1a;
        border: 1px solid #3a3a3a;
        color: #ffffff;
    }
    
    QPushButton:pressed {
        background-color: #0a0a0a;
        border: 1px solid #1a1a1a;
    }
    
    QPushButton:disabled {
        background-color: transparent;
        color: #404040;
        border: 1px solid #1a1a1a;
    }
    
    /* Minimal separator lines */
    QFrame {
        border: none;
        background-color: #1a1a1a;
    }
"""