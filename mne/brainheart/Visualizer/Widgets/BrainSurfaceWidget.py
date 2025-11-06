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

        self.hem_actors = {}
        self.electrode_actors = {}

        self.pial_plotting_params = dict(
            cmap = "gray", 
            smooth_shading = True, 
            scalars = "curvature", 
            clim = (-0.5, 0.5)
        )

        self.set_ui()
        self.load_data()



    def set_ui(self):

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        self.setLayout(layout)

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
                mesh.point_data["curvature"] = nib.freesurfer.io.read_morph_data(
                    os.path.join(surf_path, f"{hem}.curv")
                )
            self.plotter.add_mesh(
                mesh, 
                **self.pial_plotting_params
                )

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
    browser.show()