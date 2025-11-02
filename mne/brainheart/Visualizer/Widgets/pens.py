"""
Centralized pen and brush definitions for consistent styling across widgets.
Arch-inspired minimal color scheme.
"""

import pyqtgraph as pg
from PyQt5.QtCore import Qt

# ============================================================================
# Color Palette (Nord-inspired, Arch-friendly)
# ============================================================================

class Colors:
    """Color definitions for the application"""
    # Backgrounds
    MAIN_BG = '#0d0d0d'
    WIDGET_BG = '#1a1a1a'
    BORDER = '#2a2a2a'
    
    # Text
    TEXT_PRIMARY = '#e0e0e0'
    TEXT_SECONDARY = '#a0a0a0'
    TEXT_DISABLED = '#404040'
    
    # Accents
    BLUE = '#88c0d0'      # Cyan-blue
    RED = '#bf616a'       # Red
    GREEN = '#a3be8c'     # Green
    YELLOW = '#ebcb8b'    # Yellow
    PURPLE = '#b48ead'    # Purple
    ORANGE = '#d08770'    # Orange
    TEAL = '#5e81ac'      # Deep blue
    
    # ECG Signal Colors
    ECG_RAW = '#81a1c1'           # Light blue - raw signal
    ECG_CLEAN = '#88c0d0'         # Cyan - cleaned signal
    ECG_RATE = '#a3be8c'          # Green - heart rate
    ECG_QUALITY = '#ebcb8b'       # Yellow - quality metric
    
    # Peak/Feature Colors
    PEAK_R = '#bf616a'            # Red - R peaks (most important)
    PEAK_P = '#b48ead'            # Purple - P peaks
    PEAK_Q = '#d08770'            # Orange - Q peaks
    PEAK_S = '#5e81ac'            # Deep blue - S peaks
    PEAK_T = '#a3be8c'            # Green - T peaks
    
    # Onset/Offset Colors
    ONSET = '#8fbcbb'             # Light teal - onsets
    OFFSET = '#88c0d0'            # Cyan - offsets
    
    # Phase Colors
    PHASE_ATRIAL = '#b48ead'      # Purple - atrial
    PHASE_VENTRICULAR = '#bf616a' # Red - ventricular
    
    # Correction/Artifact Colors
    CORRECTED = '#a3be8c'         # Green - corrected signal
    ECTOPIC = '#bf616a'           # Red - ectopic beats
    MISSED = '#d08770'            # Orange - missed beats
    EXTRA = '#ebcb8b'             # Yellow - extra beats
    LONGSHORT = '#b48ead'         # Purple - long/short intervals
    
    # Annotations
    ANNOT_BEAT = '#a3be8c'        # Green for normal beats
    ANNOT_ABNORMAL = '#bf616a'    # Red for abnormal
    ANNOT_ARTIFACT = '#d08770'    # Orange for artifacts

# ============================================================================
# Pen Definitions
# ============================================================================

class Pens:
    """Pre-defined pens for various plot elements"""
    
    # ---- ECG Signal Types ----
    ECG_RAW = pg.mkPen(color=Colors.ECG_RAW, width=1.5)
    ECG_CLEAN = pg.mkPen(color=Colors.ECG_CLEAN, width=1.5)
    ECG_RATE = pg.mkPen(color=Colors.ECG_RATE, width=1.5)
    ECG_RATE_CORRECTED = pg.mkPen(color=Colors.CORRECTED, width=1.5, style=Qt.DashLine)
    ECG_QUALITY = pg.mkPen(color=Colors.ECG_QUALITY, width=1.2)
    
    # ---- Peak Markers ----
    PEAK_R = pg.mkPen(color=Colors.PEAK_R, width=2)
    PEAK_R_CORRECTED = pg.mkPen(color=Colors.CORRECTED, width=2, style=Qt.DashLine)
    PEAK_P = pg.mkPen(color=Colors.PEAK_P, width=1.5)
    PEAK_Q = pg.mkPen(color=Colors.PEAK_Q, width=1.5)
    PEAK_S = pg.mkPen(color=Colors.PEAK_S, width=1.5)
    PEAK_T = pg.mkPen(color=Colors.PEAK_T, width=1.5)
    
    # ---- Onset/Offset Markers ----
    ONSET = pg.mkPen(color=Colors.ONSET, width=1.2, style=Qt.DotLine)
    OFFSET = pg.mkPen(color=Colors.OFFSET, width=1.2, style=Qt.DotLine)
    
    # ---- Phase Markers ----
    PHASE_ATRIAL = pg.mkPen(color=Colors.PHASE_ATRIAL, width=1.5)
    PHASE_VENTRICULAR = pg.mkPen(color=Colors.PHASE_VENTRICULAR, width=1.5)
    
    # ---- Artifact/Correction Markers ----
    ECTOPIC = pg.mkPen(color=Colors.ECTOPIC, width=2, style=Qt.DashLine)
    MISSED = pg.mkPen(color=Colors.MISSED, width=2, style=Qt.DashLine)
    EXTRA = pg.mkPen(color=Colors.EXTRA, width=2, style=Qt.DashLine)
    LONGSHORT = pg.mkPen(color=Colors.LONGSHORT, width=2, style=Qt.DashLine)
    
    # ---- Generic ----
    TIME_TRACE = pg.mkPen(color=Colors.BLUE, width=1.0)
    POWER_SPECTRUM = pg.mkPen(color=Colors.BLUE, width=1.5)
    AXIS = pg.mkPen(color=Colors.BORDER, width=1)
    GRID = pg.mkPen(color=Colors.BORDER, width=0.5, style=Qt.DotLine)
    
    # ---- Annotations ----
    ANNOT_NORMAL = pg.mkPen(color=Colors.ANNOT_BEAT, width=2, style=Qt.DashLine)
    ANNOT_ABNORMAL = pg.mkPen(color=Colors.ANNOT_ABNORMAL, width=2, style=Qt.DashLine)
    ANNOT_ARTIFACT = pg.mkPen(color=Colors.ANNOT_ARTIFACT, width=2, style=Qt.DashLine)
    ANNOT_DEFAULT = pg.mkPen(color=Colors.RED, width=2, style=Qt.DashLine)

# ============================================================================
# Brushes for Scatter Plots (Peak Markers)
# ============================================================================

class Brushes:
    """Pre-defined brushes for filled regions and scatter points"""
    
    # Peak markers (for scatter plots)
    PEAK_R = pg.mkBrush(191, 97, 106, 200)           # Red
    PEAK_R_CORRECTED = pg.mkBrush(163, 190, 140, 200) # Green
    PEAK_P = pg.mkBrush(180, 142, 173, 200)          # Purple
    PEAK_Q = pg.mkBrush(208, 135, 112, 200)          # Orange
    PEAK_S = pg.mkBrush(94, 129, 172, 200)           # Deep blue
    PEAK_T = pg.mkBrush(163, 190, 140, 200)          # Green
    
    # Artifact markers
    ECTOPIC = pg.mkBrush(191, 97, 106, 150)          # Red
    MISSED = pg.mkBrush(208, 135, 112, 150)          # Orange
    EXTRA = pg.mkBrush(235, 203, 139, 150)           # Yellow
    LONGSHORT = pg.mkBrush(180, 142, 173, 150)       # Purple
    
    # Annotation regions (semi-transparent)
    ANNOT_NORMAL = pg.mkBrush(163, 190, 140, 25)     # Green
    ANNOT_ABNORMAL = pg.mkBrush(191, 97, 106, 25)    # Red
    ANNOT_ARTIFACT = pg.mkBrush(208, 135, 112, 25)   # Orange
    ANNOT_DEFAULT = pg.mkBrush(191, 97, 106, 25)     # Red
    
    # Confidence intervals
    CONFIDENCE_FILL = pg.mkBrush(136, 192, 208, 20)  # Blue, very transparent
    
    # Phase regions
    PHASE_ATRIAL = pg.mkBrush(180, 142, 173, 30)     # Purple
    PHASE_VENTRICULAR = pg.mkBrush(191, 97, 106, 30) # Red

# ============================================================================
# Symbols for Scatter Plots
# ============================================================================

class Symbols:
    """Symbols for different peak types"""
    PEAK_R = 'o'           # Circle
    PEAK_P = 't'           # Triangle up
    PEAK_Q = 't1'          # Triangle down
    PEAK_S = 's'           # Square
    PEAK_T = 'd'           # Diamond
    ECTOPIC = 'x'          # X mark
    MISSED = '+'           # Plus
    EXTRA = 'star'         # Star
    ONSET = '|'            # Vertical line
    OFFSET = '|'           # Vertical line

# ============================================================================
# Label Options for InfiniteLine
# ============================================================================

class LabelOpts:
    """Pre-defined label options for annotations"""
    
    NORMAL = {'position': 0.95, 'color': Colors.ANNOT_BEAT}
    ABNORMAL = {'position': 0.95, 'color': Colors.ANNOT_ABNORMAL}
    ARTIFACT = {'position': 0.95, 'color': Colors.ANNOT_ARTIFACT}
    DEFAULT = {'position': 0.95, 'color': Colors.RED}

# ============================================================================
# ECG Channel Mapping
# ============================================================================

ECG_CHANNEL_STYLES = {
    # Main signals
    'ECG_Raw': {'pen': Pens.ECG_RAW, 'type': 'line'},
    'ECG_Clean': {'pen': Pens.ECG_CLEAN, 'type': 'line'},
    'ECG_Rate': {'pen': Pens.ECG_RATE, 'type': 'line'},
    'ECG_Rate_Corrected': {'pen': Pens.ECG_RATE_CORRECTED, 'type': 'line'},
    'ECG_Quality': {'pen': Pens.ECG_QUALITY, 'type': 'line'},
    
    # R peaks (most important)
    'ECG_R_Peaks': {
        'pen': Pens.PEAK_R,
        'brush': Brushes.PEAK_R,
        'symbol': Symbols.PEAK_R,
        'size': 10,
        'type': 'scatter'
    },
    'ECG_R_Peaks_Corrected': {
        'pen': Pens.PEAK_R_CORRECTED,
        'brush': Brushes.PEAK_R_CORRECTED,
        'symbol': Symbols.PEAK_R,
        'size': 10,
        'type': 'scatter'
    },
    
    # Other peaks
    'ECG_P_Peaks': {
        'pen': Pens.PEAK_P,
        'brush': Brushes.PEAK_P,
        'symbol': Symbols.PEAK_P,
        'size': 8,
        'type': 'scatter'
    },
    'ECG_Q_Peaks': {
        'pen': Pens.PEAK_Q,
        'brush': Brushes.PEAK_Q,
        'symbol': Symbols.PEAK_Q,
        'size': 8,
        'type': 'scatter'
    },
    'ECG_S_Peaks': {
        'pen': Pens.PEAK_S,
        'brush': Brushes.PEAK_S,
        'symbol': Symbols.PEAK_S,
        'size': 8,
        'type': 'scatter'
    },
    'ECG_T_Peaks': {
        'pen': Pens.PEAK_T,
        'brush': Brushes.PEAK_T,
        'symbol': Symbols.PEAK_T,
        'size': 8,
        'type': 'scatter'
    },
    
    # Onsets/Offsets
    'ECG_P_Onsets': {
        'pen': Pens.ONSET,
        'brush': Brushes.PEAK_P,
        'symbol': Symbols.ONSET,
        'size': 6,
        'type': 'scatter'
    },
    'ECG_P_Offsets': {
        'pen': Pens.OFFSET,
        'brush': Brushes.PEAK_P,
        'symbol': Symbols.OFFSET,
        'size': 6,
        'type': 'scatter'
    },
    'ECG_R_Onsets': {
        'pen': Pens.ONSET,
        'brush': Brushes.PEAK_R,
        'symbol': Symbols.ONSET,
        'size': 6,
        'type': 'scatter'
    },
    'ECG_R_Offsets': {
        'pen': Pens.OFFSET,
        'brush': Brushes.PEAK_R,
        'symbol': Symbols.OFFSET,
        'size': 6,
        'type': 'scatter'
    },
    'ECG_T_Onsets': {
        'pen': Pens.ONSET,
        'brush': Brushes.PEAK_T,
        'symbol': Symbols.ONSET,
        'size': 6,
        'type': 'scatter'
    },
    'ECG_T_Offsets': {
        'pen': Pens.OFFSET,
        'brush': Brushes.PEAK_T,
        'symbol': Symbols.OFFSET,
        'size': 6,
        'type': 'scatter'
    },
    
    # Artifact markers
    'ECG_fixpeaks_ectopic': {
        'pen': Pens.ECTOPIC,
        'brush': Brushes.ECTOPIC,
        'symbol': Symbols.ECTOPIC,
        'size': 12,
        'type': 'scatter'
    },
    'ECG_fixpeaks_missed': {
        'pen': Pens.MISSED,
        'brush': Brushes.MISSED,
        'symbol': Symbols.MISSED,
        'size': 12,
        'type': 'scatter'
    },
    'ECG_fixpeaks_extra': {
        'pen': Pens.EXTRA,
        'brush': Brushes.EXTRA,
        'symbol': Symbols.EXTRA,
        'size': 12,
        'type': 'scatter'
    },
    'ECG_fixpeaks_longshort': {
        'pen': Pens.LONGSHORT,
        'brush': Brushes.LONGSHORT,
        'symbol': Symbols.EXTRA,
        'size': 10,
        'type': 'scatter'
    },
    
    # Phase signals
    'ECG_Phase_Atrial': {'pen': Pens.PHASE_ATRIAL, 'type': 'line'},
    'ECG_Phase_Completion_Atrial': {'pen': Pens.PHASE_ATRIAL, 'type': 'line'},
    'ECG_Phase_Ventricular': {'pen': Pens.PHASE_VENTRICULAR, 'type': 'line'},
    'ECG_Phase_Completion_Ventricular': {'pen': Pens.PHASE_VENTRICULAR, 'type': 'line'},
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_ecg_style(channel_name: str) -> dict:
    """
    Get the appropriate style for an ECG channel.
    
    Parameters
    ----------
    channel_name : str
        Name of the ECG channel (e.g., 'ECG_Raw', 'ECG_R_Peaks', etc.)
    
    Returns
    -------
    style : dict
        Dictionary with styling information
    """
    return ECG_CHANNEL_STYLES.get(
        channel_name,
        {'pen': Pens.ECG_RAW, 'type': 'line'}  # Default style
    )

def is_peak_channel(channel_name: str) -> bool:
    """Check if channel represents peaks (discrete events)"""
    return 'Peaks' in channel_name or 'fixpeaks' in channel_name or \
           'Onsets' in channel_name or 'Offsets' in channel_name

def is_continuous_channel(channel_name: str) -> bool:
    """Check if channel represents continuous signal"""
    continuous_keywords = ['Raw', 'Clean', 'Rate', 'Quality', 'Phase']
    return any(keyword in channel_name for keyword in continuous_keywords)

def get_annotation_style(annotation_type: str):
    """
    Get pen, brush, and label options for an annotation type.
    
    Parameters
    ----------
    annotation_type : str
        Type of annotation
    
    Returns
    -------
    style : dict
        Dictionary with 'pen', 'brush', and 'label_opts' keys
    """
    annotation_type = annotation_type.lower()
    
    if 'normal' in annotation_type or 'beat' in annotation_type:
        return {
            'pen': Pens.ANNOT_NORMAL,
            'brush': Brushes.ANNOT_NORMAL,
            'label_opts': LabelOpts.NORMAL
        }
    elif 'abnormal' in annotation_type or 'arrhythmia' in annotation_type:
        return {
            'pen': Pens.ANNOT_ABNORMAL,
            'brush': Brushes.ANNOT_ABNORMAL,
            'label_opts': LabelOpts.ABNORMAL
        }
    elif 'artifact' in annotation_type or 'noise' in annotation_type:
        return {
            'pen': Pens.ANNOT_ARTIFACT,
            'brush': Brushes.ANNOT_ARTIFACT,
            'label_opts': LabelOpts.ARTIFACT
        }
    else:
        return {
            'pen': Pens.ANNOT_DEFAULT,
            'brush': Brushes.ANNOT_DEFAULT,
            'label_opts': LabelOpts.DEFAULT
        }