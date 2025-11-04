import numpy as np
import mne

import matplotlib.pyplot as plt

from functools import partial

from mne.brainheart.Visualizer.MRI_Visualizer import (
    create_mri_plot, 
    update_mri_plot
)

import pyvista as pv

mne.viz.set_3d_backend('pyvista')
import matplotlib
matplotlib.use("Qt5Agg")
plt.ion()

def setup_mri_picking(
        brain, 
        info, 
        t1): 
    
    if not brain._surf == "pial": 
        raise ValueError("Enter the Pial Data")
    
    if brain._units == "mm": 
        click_pos_adj = 1000
    elif brain._units == "m": 
        click_pos_adj = 1
    else: 
        raise Warning("Unknown Units for Brain Coords")
    
    def find_nearest_electrode(click_pos): 
        click_pos = np.array(click_pos)/click_pos_adj
        min_distance = np.inf
        nearest_electrode = None
        nearest_pos = None
        for ch in info["chs"]: 
            if ch["kind"] == 802: 
                electrode_pos = ch["loc"][:3]
                distance = np.linalg.norm(click_pos - np.array(electrode_pos))
                if distance < min_distance: 
                    min_distance = distance
                    nearest_electrode = ch["ch_name"]
                    nearest_pos = electrode_pos
        return nearest_electrode, np.array(nearest_pos)*click_pos_adj
    

    def click_callback(point): 
        print(f"Clicked at 3D position: {point}")
        electrode_name, nearest_pos = find_nearest_electrode(point)
        update_mri_plot_partial(
            point = nearest_pos, 
            electrode_name = electrode_name, 
        )
        update_selected_electrode(nearest_pos, electrode_name)
        if electrode_name is not None: 
            print(f"Nearest Electrode: {electrode_name}")
        else: 
            print(f"No electrode nearby")


    def add_name_labels(info, plotter):
        electrode_positions = []
        electrode_names = [] 
        for ch in info["chs"]: 
            if ch["kind"] == 802: 
                electrode_positions.append(np.array(ch["loc"][:3]*click_pos_adj))
                electrode_names.append(ch["ch_name"])

        if electrode_positions: 
            plotter.add_point_labels(
                points = electrode_positions, 
                labels = electrode_names, 
                point_size = 0, 
                font_size = 10, 
                text_color = "yellow", 
                shape_color = "black", 
                shape_opacity = 0.8, 
                name = "all_electrode_labels"
            ) 

    data = t1.get_fdata()

    affine = t1.affine #voxel -> Anat
    affine_inv = np.linalg.inv(affine) #Anat -> Voxel

    fig, axes, vmin, vmax = create_mri_plot(data)
    kwargs = dict(
        affine_inv = affine_inv, 
        axes = axes, 
        data = data, 
        fig = fig, 
        vmin = vmin, 
        vmax = vmax, 
    )
    update_mri_plot_partial = partial(
        update_mri_plot, **kwargs
    )
    update_mri_plot_partial(
        point = (0, 0, 0),
        electrode_name = None
    )
    plotter = brain._renderer.plotter

    for actor_name, actor in plotter.renderer.actors.items(): 
        actor_name_str = str(actor_name).lower()
        print(actor_name_str)
        if "sensor" in actor_name_str or "electrode" in actor_name_str: 
            actor.setPickable(True)
            print(f"Set {actor_name} to pickable")
    
    print("\nMaking brain surfaces non-pickable...")
    for hemi in ["lh", "rh"]: 
        if hemi in brain._layered_meshes: 
            layered_mesh = brain._layered_meshes[hemi]
            if hasattr(layered_mesh, "_actor"): 
                layered_mesh._actor.SetPickable(False)
                print(f"   Set {hemi} brain surface to non-pickable")

    add_name_labels(info, plotter)

    # Now setup the current picked electrode
    def update_selected_electrode(point, electrode_name): 
        # Temporary Fix until something better is figured out
        wash_curr_selected_electrode()
        draw_curr_selected_electrode(point, electrode_name)
    
    def draw_curr_selected_electrode(point, electrode_name):
        print(f"Adding Point{point}")
        marker = pv.Sphere(radius=3, center=point)
        actor = plotter.add_mesh(
            marker,
            color='blue',  # or 'cyan', 'magenta', etc.
            opacity=0.5,
            style='surface',  # or 'surface', 'points'
            line_width=3,
            name='selection_marker',
            pickable=False
        )
        plotter.render()
        brain.curr_selected_electrode = actor
    
    def wash_curr_selected_electrode(): 
        if not hasattr(brain, "curr_selected_electrode") or brain.curr_selected_electrode is None: 
            return
        plotter.remove_actor(brain.curr_selected_electrode)
        brain.curr_selected_electrode = None
        

    # plotter.add_key_event()
    plotter.track_click_position(callback = click_callback)
    plotter.disable_picking()
    # Better Rotation Style
    plotter.enable_trackball_style()

    return fig, axes


if __name__ == "__main__": 
    
    import mne_bids
    import nibabel as nb
    bids_root = r"D:/DABI/StimulationDataset"
    ext = "vhdr" #extension for the recording
    subject = "0d9l" #sample
    sess = "postimp"
    datatype = "ieeg"
    suffix = "ieeg"
    run = "01"
    extension = "vhdr"
    bids_paths = mne_bids.BIDSPath(root = bids_root, 
                                session = sess, 
                                subject = subject, 
                                datatype=datatype, 
                                suffix = suffix,
                                run = run, 
                                extension= extension
                                )
    bids_path = bids_paths.match()[0]
    #Load
    raw = mne_bids.read_raw_bids(bids_path)

    brain = mne.viz.Brain(
        f"sub-{subject}",
        subjects_dir=r"D:\DABI\StimulationDataset\derivatives\freesurfer",
        alpha = 0.7, 
        show = False, 
        surf = "pial")
    # To get the transform
    path = r"D:\DABI\StimulationDataset\derivatives\freesurfer\sub-0d9l\surf\ct.mgz"
    import nibabel as nib
    mri = nib.load(path)
    trans = mri.header.get_vox2ras_tkr()
    trans = mne.transforms.Transform(fro="head", to="mri", trans=np.eye(4))
    brain.add_sensors(raw.info, trans=trans, seeg = True)
    brain.add_annotation("aparc", borders = False, alpha = 0.2)
    info = raw.info
    t1 = nb.load(rf"D:\DABI\StimulationDataset\sub-{subject}\ses-preimp\anat\sub-{subject}_ses-preimp_acq-T1w_run-01_T1w.nii")
    mri_fig, mri_axes = setup_mri_picking(brain, info, t1)
    brain._renderer.plotter.enable_surface_point_picking(False)
    brain.show()

    print("Click on the brain to update MRI slices. Close the brain window to exit")
    input("Press Enter to exit...")


'''# Option 1: Force everything into one event loop
import threading
def threaded_update(point):
    threading.Thread(target=update_mri_plot, args=(point,), daemon=True).start()

# Option 2: Use Qt directly for both
from PyQt5 import QtWidgets
# Create native Qt windows for everything

# Option 3: Use matplotlib's animation framework
from matplotlib.animation import FuncAnimation
# Set up continuous updating animation

# Option 4: Use notebook backend for everything
mne.viz.set_3d_backend('notebook')
# Keep everything in Jupyter environment'''