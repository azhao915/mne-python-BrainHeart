import numpy as np
from PyQt5.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QLabel, 
    QGraphicsView, 
    QGraphicsScene, 
    QGraphicsPixmapItem, 
    QGraphicsLineItem
)

from PyQt5.QtGui import (
    QImage, 
    QPixmap, 
    QPen, 
    QClipboard, 
    QFont
)
    
from PyQt5.QtCore import (
    Qt, 
    QLineF
)

from nilearn.image.resampling import reorder_img

class MRISliceVIewer(QGraphicsView): 
    def __init__(
            self, 
            slice_name: str = "", 
            parent = None): 
        super().__init__(parent)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.slice_name = slice_name

        self.xline, self.yline = None, None
        self.text_items = []

        self._image_data = None
        '''
        self.setStyle(
            "background-color: black; border: 1px solid gray;"
        )
        '''
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.pixmap_item = None


    def display_slice(
            self, 
            slice_data, 
            vmin, 
            vmax,
            marker_line_pos = None, 
            chan_name: str = ""
    ): 
        self.scene.clear()
        normalized_data = np.clip(
            (slice_data - vmin) / (vmax - vmin) * 255, 0, 255
        )
        normalized_data = np.round(normalized_data).astype(np.uint8)
        if not normalized_data.flags['C_CONTIGUOUS']:
            normalized_data = np.ascontiguousarray(normalized_data)
        self._image_data = normalized_data
        height, width = normalized_data.shape
        q_image = QImage(normalized_data.tobytes(), width, height, width, 
                        QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
    
    def resizeEvent(self, event): 
        super().resizeEvent(event)
        if self.pixmap_item is not None: 
            self.fitInView(
                self.scene.sceneRect(), 
                Qt.KeepAspectRatio
            )

class MRIViewer(QWidget): 
    def __init__(
            self, 
            t1,
            parent = None): 
        super().__init__(parent)
        self.setup_data(t1)
        self.setup_ui()

    def setup_data(
            self, 
            t1
    ): 
        print(t1.affine)
        t1 = reorder_img(t1)
        print(t1.affine)
        data, affine = t1.get_fdata(), t1.affine
        affine_inv = np.linalg.inv(affine)
        self.data = data,
        self.affine_inv = affine_inv

        self.vmin = np.min(data)
        self.vmax = np.max(data)
        dim = 0.6
        vmean = 0.5 * (self.vmin + self.vmax)
        ptp = 0.5 * (self.vmax - self.vmin)
        self.vmax = vmean + (1 + dim) * ptp

    def setup_ui(
        self
    ): 
        self.coronal_view = MRISliceVIewer("Coronal")
        self.sagittal_view = MRISliceVIewer("Sagittal")
        self.horizontal_view = MRISliceVIewer("Horizontal")

        layout = QHBoxLayout()
        layout.addWidget(self.coronal_view)
        layout.addWidget(self.sagittal_view)
        layout.addWidget(self.horizontal_view)

        self.setLayout(layout)
        self.setStyleSheet("background-color: black;")

        self._update_display((0, 0, 0), electrode_name=None)  

    def _update_display(
            self, 
            point, 
            electrode_name = None
    ): 
        if self.data is None:
            return
        coords_vox_indices = self.point_to_voxels(point)

        if isinstance(self.data, tuple): 
            self.data = self.data[0]
        
        coronal_slice = self.data[:, coords_vox_indices[1], ::-1].T
        sagittal_slice = self.data[coords_vox_indices[0], :, ::-1].T
        horizontal_slice = self.data[:, ::-1, coords_vox_indices[2]].T

        self.coronal_view.display_slice(coronal_slice, self.vmin, self.vmax)
        self.sagittal_view.display_slice(sagittal_slice, self.vmin, self.vmax)
        self.horizontal_view.display_slice(horizontal_slice, self.vmin, self.vmax)

    def point_to_voxels(self, point): 
        
        coords = list(point)
        coords_anat = np.array((coords + [1]))
        coords_vox = self.affine_inv @ coords_anat
        coords_vox_indices = np.round(coords_vox).astype(int)[:3]
        '''
        for coord_index in range(3): 
            coords_vox_indices[coord_index] = np.clip(coords_vox_indices[coord_index], 0, self.data.shape[coord_index]-1)
        '''
        return coords_vox_indices


# Test it
if __name__ == "__main__": 
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    subject = "4r3o"
    import nibabel as nib
    t1 = nib.load(rf"D:\DABI\StimulationDataset\sub-{subject}\ses-preimp\anat\sub-{subject}_ses-preimp_acq-T1w_run-01_T1w.nii")
    widget = MRIViewer(t1)
    main_window.setCentralWidget(
        widget
    )
    widget._update_display(
        (0, 0, 0)
    )
    main_window.show()

