import nibabel as nib
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, 
    QWidget, 
    QVBoxLayout
)
from pyvistaqt import QtInteractor
import pyvista as pv

import os

class BrainSurfaceWidget(QWidget): 
    def __init__(
            self, 
            subject: str, 
            freesurfer_path: str,
            parent = None): 
        super().__init__(parent)

        self.subject_path = os.path.join(
            freesurfer_path, subject
        )

        self.show_labels = False

        self.sensors = {}

        self.pial_plotting_params = dict(                
            cmap=['#D3D3D3', '#808080'],
            clim=[0.1, 0.2],  # Binary range
            opacity=0.3,
            smooth_shading=True,
            show_scalar_bar=False,
            lighting=True,
            interpolate_before_map=True
        )

        self.click_pos_adj = 1000

        self.set_ui()
        self.load_data()



    def set_ui(self):

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        self.setLayout(layout)

        self.plotter.set_background("black")

    def load_data(self): 
        surf_path = os.path.join(self.subject_path, "surf")
        for hem in ["lh", "rh"]: 
            vertices, triangles = nib.freesurfer.io.read_geometry(
                os.path.join(surf_path, f"{hem}.pial")
            )
            faces = self.triangles_as_faces(triangles)
            mesh = pv.PolyData(vertices, faces)
            # Now load the Curvature Values
            if f"{hem}.curv" in os.listdir(surf_path): 
                curv = nib.freesurfer.io.read_morph_data(
                    os.path.join(surf_path, f"{hem}.curv")
                )
                bin_curv = (curv > 0).astype(float)
                mesh.point_data["curvature"] = bin_curv
            actor = self.plotter.add_mesh(
                mesh, 
                scalars = "curvature", 
                **self.pial_plotting_params
                )
            actor.SetPickable(False)

        self.curr_selected_sensor = None
            
    
    def add_sensors(self, info): 
        positions = []
        names = []

        for ch in info["chs"]: 
            if ch["kind"] != 802:
                # Not sEEG
                continue
            name, pos = ch["ch_name"], ch["loc"][:3]*self.click_pos_adj
            ch_info = {
                "pos": pos, 
                "index": len(positions)
            }
            self.sensors[name] = ch_info
            positions.append(pos)
            names.append(name)

        self.sensor_positions = np.array(positions)
        self.sensor_names = names

        self.plot_sensors()
        self.plot_sensor_labels()

    def plot_sensors(
        self
    ): 
        if not len(self.sensor_positions): 
            return 
        self.sensor_mesh = pv.PolyData(self.sensor_positions)
        self.sensor_actor = self.plotter.add_mesh(
            self.sensor_mesh, 
            point_size = 10, 
            color = "cyan",
            render_points_as_spheres = True,
            name = "all_sensors", 
            pickable = True
        )

    def plot_sensor_labels(
            self, 
            offset = 1
    ): 
        if not len(self.sensor_positions): 
            return 
        if not self.show_labels: 
            return
        self.sensor_text_mesh = pv.PolyData(self.sensor_positions + offset)
        self.sensor_label_actor = self.plotter.add_point_labels(
            self.sensor_text_mesh,
            self.sensor_names,
            point_size=0,
            font_size=8,
            text_color='cyan',
            render_points_as_spheres=True,
            always_visible=False,
            name="all_sensors"
        )

    
    def highlight_electrode(self, index): 
        self.clear_highlight()
        
        pos = self.sensor_positions[index, :]
        curr_selected_data = pv.PolyData(pos)
        
        self.curr_selected_sensor = self.plotter.add_mesh(
            curr_selected_data, 
            point_size = 15,
            color = "red", 
            render_points_as_spheres = True, 
            pickable = True
        )

    def clear_highlight(self): 
        if self.curr_selected_sensor is None: 
            return
        self.plotter.remove_actor(self.curr_selected_sensor)


    def triangles_as_faces(self, triangles): 
        n_triangles = triangles.shape[0]
        faces = np.column_stack([
            np.full(n_triangles, 3),  
            triangles
        ]).ravel()
        return faces


# First do a little test
if __name__ == "__main__": 

    import sys
    app = QApplication(sys.argv)

    freesurf_path = r"D:\DABI"
    subject = "sub-4r3o"
    browser = BrainSurfaceWidget(subject, freesurf_path)

    from mne.brainheart.load_reference_dataset import load
    raw = load()
    browser.add_sensors(raw.info)
    browser.show()