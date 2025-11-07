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

freqs = np.arange(50) + 2
picks = np.arange(10) + 65

tf = raw.copy().pick(picks).compute_tfr(method = "morlet", freqs = freqs, decim = 100)

tfr_browser = TFRBrowser(
    tf = tf,
    raw = raw)

#############
# MRI Slice Viewer
#############
subject = "sub-4r3o"

t1 = nib.load(r"D:\DABI\sub-4r3o\mri\T1.mgz")
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