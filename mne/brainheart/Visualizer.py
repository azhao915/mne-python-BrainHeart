import numpy as np
import mne

import matplotlib.pyplot as plt

def setup_simple_electrode_picking(
        brain, 
        info, 
        min_distance: float = 1, 
        click_pos_adj: int = 1000): 
    

    def find_nearest_electrode(click_pos): 
        click_pos = np.array(click_pos)/click_pos_adj
        min_distance = np.inf
        nearest_electrode = None
        for ch in info["chs"]: 
            if ch["kind"] == 802: 
                electrode_post = ch["loc"][:3]
                distance = np.linalg.norm(click_pos - np.array(electrode_post))
                if distance < min_distance: 
                    min_distance = distance
                    nearest_electrode = ch["ch_name"]
        return nearest_electrode


    def click_callback(point): 
        print(f"Clicked at 3D position: {point}")
        electrode_name = find_nearest_electrode(point)
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
    
    plotter = brain._renderer.plotter
    add_name_labels(info, plotter)
    # plotter.add_key_event()
    plotter.track_click_position(callback = click_callback)
    # Better Rotation Style
    plotter.enable_trackball_style()

def setup_mri_picking(
        brain, 
        info, 
        t1, 
        min_distance: float = 1, 
        click_pos_adj: int = 1000): 
    

    def find_nearest_electrode(click_pos): 
        click_pos = np.array(click_pos)/click_pos_adj
        min_distance = np.inf
        nearest_electrode = None
        for ch in info["chs"]: 
            if ch["kind"] == 802: 
                electrode_post = ch["loc"][:3]
                distance = np.linalg.norm(click_pos - np.array(electrode_post))
                if distance < min_distance: 
                    min_distance = distance
                    nearest_electrode = ch["ch_name"]
        return nearest_electrode


    def click_callback(point): 
        print(f"Clicked at 3D position: {point}")
        electrode_name = find_nearest_electrode(point)
        update_mri_plot(point)
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
    

    def create_mri_plot(): 
        fig, axes = plt.subplots(1, 3, figsize = (10, 5))
        vmin = np.min(data)
        vmax = np.max(data)
        dim = 0.6

        vmean = 0.5 * (vmin + vmax)
        ptp = 0.5 * (vmax - vmin)
        vmax = vmean + (1 + dim) * ptp

        axes[0].set_title("Coronal", color = "white")
        axes[1].set_title("Sagittal", color = "white")
        axes[2].set_title("Axial", color = "white")

        for ax in axes: 
            ax.set_xticks([])
            ax.set_yticks([])
        
        fig.set_facecolor("black")

        plt.show(block = False)
        plt.pause(0.1)

        return fig, axes, vmin, vmax


    def update_mri_plot(point): 
        print(f"Updating MRI, point = {point}")
        # TO DO: Need to convert make sure the position is correct
        coords = list(point)
        coords_anat = np.array((coords + [1]))
        coords_vox = affine_inv @ coords_anat
        coords_vox_indices = coords_vox.astype(int)[:3]

        for coord_index in range(3): 
            coords_vox_indices[coord_index] = np.clip(coords_vox_indices[coord_index], 0, data.shape[coord_index]-1)

        cmap = "grey"

        for ax in axes:
            ax.clear()

        coronal_slice = data[::-1, :, coords_vox_indices[2]].T
        sagittal_slice = data[coords_vox_indices[0], :, :]
        horizontal_slice = data[::-1, coords_vox_indices[1], ::-1].T

        axes[0].imshow(coronal_slice, cmap = cmap, vmin = vmin, vmax = vmax)
        axes[0].axhline(coords_vox_indices[1], color = "white")
        axes[0].axvline(data.shape[0]-coords_vox_indices[0], color = "white")
        axes[0].text(0.05, 0.95, 'L', transform=axes[0].transAxes, 
                    color='white', fontsize=14, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='left')
        axes[0].text(0.95, 0.95, 'R', transform=axes[0].transAxes, 
                    color='white', fontsize=14, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='right')
        axes[0].text(0.05, 0.05, f"y = {coords_vox_indices[2]}", transform=axes[0].transAxes, 
                    color='white', fontsize=14, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='left')

        im1 = axes[1].imshow(sagittal_slice, cmap = cmap, vmin = vmin, vmax = vmax)
        axes[1].axhline(coords_vox_indices[1], color = "white")
        axes[1].axvline(coords_vox_indices[2], color = "white")
        axes[1].text(0.05, 0.05, f"x = {coords_vox_indices[0]}", transform=axes[1].transAxes, 
                    color='white', fontsize=14, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='left')

        axes[2].imshow(horizontal_slice, cmap = cmap, vmin = vmin, vmax = vmax)
        axes[2].axhline(data.shape[2] - coords_vox_indices[2], color = "white")
        axes[2].axvline(data.shape[0] - coords_vox_indices[0], color = "white")
        axes[2].text(0.05, 0.95, 'L', transform=axes[2].transAxes, 
                    color='white', fontsize=14, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='left')
        axes[2].text(0.95, 0.95, 'R', transform=axes[2].transAxes, 
                    color='white', fontsize=14, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='right')
        axes[2].text(0.05, 0.05, f"x = {coords_vox_indices[1]}", transform=axes[2].transAxes, 
                    color='white', fontsize=14, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='left')

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        if hasattr(fig, "_colorbar") and fig._colorbar is not None: 
            fig._colorbar.remove()
        fig._colorbar = fig.colorbar(im1, ax = axes, shrink = 0.8, aspect = 20, pad = 0.02)
        fig._colorbar.ax.tick_params(colors = "white")
        
        plt.draw()
        plt.pause(0.01)

    data = t1.get_fdata()

    affine = t1.affine #voxel -> Anat
    affine_inv = np.linalg.inv(affine) #Anat -> Voxel

    fig, axes, vmin, vmax = create_mri_plot()
    update_mri_plot((0, 0, 0))
    plotter = brain._renderer.plotter
    add_name_labels(info, plotter)
    # plotter.add_key_event()
    plotter.track_click_position(callback = click_callback)
    # Better Rotation Style
    plotter.enable_trackball_style()

    return fig, axes


if __name__ == "__main__": 
    
    import mne_bids
    import nibabel as nb
    bids_root = r"D:/DABI/StimulationDataset"
    ext = "vhdr" #extension for the recording
    subject = "4r3o" #sample
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

    mne.viz.set_3d_backend('pyvista')
    import matplotlib
    matplotlib.use("Qt5Agg")
    plt.ion()

    brain = mne.viz.Brain(
        f"sub-{subject}",
        subjects_dir=r"D:\DABI\StimulationDataset\derivatives\freesurfer",
        alpha = 0.7, 
        show = False)
    trans = mne.transforms.Transform(fro="head", to="mri", trans=np.eye(4))
    brain.add_sensors(raw.info, trans=trans)
    brain.add_annotation("aparc", borders = False, alpha = 0.2)
    info = raw.info
    t1 = nb.load(r"D:\DABI\StimulationDataset\sub-4r3o\ses-preimp\anat\sub-4r3o_ses-preimp_acq-T1w_run-01_T1w.nii")
    mri_fig, mri_axes = setup_mri_picking(brain, info, t1)
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