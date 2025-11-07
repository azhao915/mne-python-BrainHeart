from PyQt5.QtWidgets import QApplication
import sys
from mne.brainheart.load_reference_dataset import load

import numpy as np

from mne.brainheart.Visualizer.Browsers.TFRBrowser import TFRBrowser

from mne.brainheart.Visualizer.Widgets.BrainSurfaceWidget import BrainSurfaceWidget
from mne.brainheart.Visualizer.Widgets.MRISliceView import MRIViewer

import nibabel as nib

app = QApplication(sys.argv)

raw = load(0)

##############
# TFR Browser
##############

from mne.time_frequency import read_tfrs

# Load Pre-computed Time-Frequency Representation
tf = read_tfrs("test_tf.h5")
tfr_browser = TFRBrowser(
    tf = tf,
    raw = raw)

#############
# MRI Slice Viewer
#############
subject = "sub-4r3o"

t1 = nib.load("D:\DABI\sub-4r3o\mri\T1.mgz")
mri_widget = MRIViewer(t1, raw.info)
tfr_browser.channel_manager.register_widget(mri_widget)

#############
# Surface Viewer
#############
freesurf_path = r"D:\DABI"
surface_widget = BrainSurfaceWidget(subject, freesurf_path)
surface_widget.add_sensors(raw.info)
surface_widget.link_channel_manager(tfr_browser.channel_manager)

tfr_browser.show()
mri_widget.show()
surface_widget.show()

sys.exit(app.exec_())