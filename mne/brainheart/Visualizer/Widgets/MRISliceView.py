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
    QColor,
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
            slice_name: str, 
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

        self.add_markers(
            marker_line_pos, 
            width, 
            height
        )

    
    def add_markers(self, marker_line_pos, width, height): 
        if marker_line_pos is None: 
            return
        pen = QPen(QColor("white"), 1)

        self.marker_horz = QGraphicsLineItem(
            QLineF(0, marker_line_pos[1], width, marker_line_pos[1])
        )
        self.marker_horz.setPen(pen)
        self.scene.addItem(self.marker_horz)

        self.marker_vert = QGraphicsLineItem(
            QLineF(marker_line_pos[0], 0, marker_line_pos[0], height)
        )
        self.marker_vert.setPen(pen)
        self.scene.addItem(self.marker_vert)

    
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
            info,
            parent = None): 
        super().__init__(parent)
        
        self.click_pos_adj = 1000

        self.setup_chs(info)
        self.setup_data(t1)
        self.setup_ui()

    def setup_chs(
            self, 
            info
    ): 
        # Taken from the Brain Surface Widget code
        positions = []
        names = []

        for ch in info["chs"]: 
            if ch["kind"] != 802:
                # Not sEEG
                continue
            name, pos = ch["ch_name"], ch["loc"][:3] * self.click_pos_adj
            positions.append(pos)
            names.append(name)

        self.sensor_positions = np.array(positions)
        self.sensor_names = names

    def setup_data(
            self, 
            t1
    ): 
        t1 = reorder_img(t1)

        vox2ras_tkr = t1.header.get_vox2ras_tkr()
        self.raw_tfr2vox = np.linalg.inv(vox2ras_tkr)

        data, affine = t1.get_fdata(), t1.affine
        affine_inv = np.linalg.inv(affine)
        self.data = data,
        #self.affine_inv = affine_inv
        self.affine_inv = self.raw_tfr2vox
        
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

        self._update_display((0, 0, 0), curr_channel_name = None)  

    def _update_display(
            self, 
            point = (0, 0, 0), 
            curr_channel_name = None
    ): 
        if self.data is None:
            return
        
        if curr_channel_name is not None and curr_channel_name in self.sensor_names: 
            chan_index = self.sensor_names.index(curr_channel_name)
            point = self.sensor_positions[chan_index, :]

        coords_vox_indices = self.point_to_voxels(point)

        if isinstance(self.data, tuple): 
            self.data = self.data[0]
        
        coronal_slice = self.data[:, coords_vox_indices[1], ::-1].T
        sagittal_slice = self.data[coords_vox_indices[0], :, ::-1].T
        horizontal_slice = self.data[:, ::-1, coords_vox_indices[2]].T

        self.coronal_view.display_slice(
            coronal_slice, 
            self.vmin, 
            self.vmax, 
            marker_line_pos = (coords_vox_indices[0], 
                               self.data.shape[2] - coords_vox_indices[2]))
        print(f"Coronal: {(coords_vox_indices[0], 
                               self.data.shape[2] - coords_vox_indices[2])}")
        self.sagittal_view.display_slice(
            sagittal_slice, 
            self.vmin, 
            self.vmax, 
            marker_line_pos = (coords_vox_indices[1], 
                               self.data.shape[2] - coords_vox_indices[2]))
        print(f"Sagittal: {(coords_vox_indices[1], 
                               self.data.shape[2] - coords_vox_indices[2])}")
        self.horizontal_view.display_slice(
            horizontal_slice, 
            self.vmin, 
            self.vmax, 
            marker_line_pos = (coords_vox_indices[0],
                          self.data.shape[1] - coords_vox_indices[1]))
        print(f"Horizonta: {(coords_vox_indices[0],
                          self.data.shape[1] - coords_vox_indices[1])}")

    def point_to_voxels(self, point): 
        print(point)
        coords = list(point)
        coords_anat = np.array((coords + [1]))
        coords_vox = self.affine_inv @ coords_anat
        coords_vox_indices = np.round(coords_vox).astype(int)[:3]
        '''
        for coord_index in range(3): 
            coords_vox_indices[coord_index] = np.clip(coords_vox_indices[coord_index], 0, self.data.shape[coord_index]-1)
        '''
        print(coords_vox_indices)
        return coords_vox_indices


# Test it
if __name__ == "__main__": 
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    app = QApplication(sys.argv)
    main_window = QMainWindow()

    subject = "4r3o"
    import nibabel as nib
    t1 = nib.load(r"D:\DABI\sub-4r3o\mri\T1.mgz")


    from mne.brainheart.load_reference_dataset import load
    
    raw = load()

    from mne.brainheart.Visualizer.Managers.ChannelManager import ChannelManager

    channel_manager = ChannelManager(raw.ch_names)

    widget = MRIViewer(t1, raw.info)
    main_window.setCentralWidget(
        widget
    )
    channel_manager.register_widget(widget)
    channel_manager.update_channel_name("RAV02")
    main_window.show()

    

