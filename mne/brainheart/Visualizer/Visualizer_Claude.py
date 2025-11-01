import numpy as np
import mne
import matplotlib.pyplot as plt
from functools import partial

from mne.brainheart.Visualizer.MRI_Visualizer import (
    create_mri_plot, 
    update_mri_plot
)

mne.viz.set_3d_backend('pyvista')
import matplotlib
matplotlib.use("Qt5Agg")
plt.ion()


def setup_mri_picking(brain, info, t1): 
    
    if brain._surf != "pial": 
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
            point=nearest_pos, 
            electrode_name=electrode_name, 
        )
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
                points=electrode_positions, 
                labels=electrode_names, 
                point_size=0, 
                font_size=8, 
                text_color="yellow", 
                shape_color="black", 
                shape_opacity=0.8, 
                name="all_electrode_labels"
            ) 

    data = t1.get_fdata()
    affine = t1.affine
    affine_inv = np.linalg.inv(affine)

    fig, axes, vmin, vmax = create_mri_plot(data)
    kwargs = dict(
        affine_inv=affine_inv, 
        axes=axes, 
        data=data, 
        fig=fig, 
        vmin=vmin, 
        vmax=vmax, 
    )
    update_mri_plot_partial = partial(
        update_mri_plot, **kwargs
    )
    update_mri_plot_partial(
        point=(0, 0, 0),
        electrode_name=None
    )
    
    plotter = brain._renderer.plotter
    
    # Make sure electrode actors ARE pickable
    print("\nMaking electrode sensors pickable...")
    for actor_name, actor in plotter.renderer.actors.items():
        actor_name_str = str(actor_name).lower()
        if 'sensor' in actor_name_str or 'electrode' in actor_name_str:
            actor.SetPickable(True)
            print(f"  Set {actor_name} to pickable")

    # KEY CHANGE: Make brain surface actors non-pickable via _layered_meshes
    print("\nMaking brain surfaces non-pickable...")
    for hemi in ['lh', 'rh']:
        if hemi in brain._layered_meshes:
            layered_mesh = brain._layered_meshes[hemi]
            if hasattr(layered_mesh, '_actor'):
                layered_mesh._actor.SetPickable(False)
                print(f"  Set {hemi} brain surface to non-pickable")
    
    add_name_labels(info, plotter)
    plotter.track_click_position(callback=click_callback)
    plotter.enable_trackball_style()

    return fig, axes


if __name__ == "__main__": 
    
    import mne_bids
    import nibabel as nb
    
    bids_root = r"D:/DABI/StimulationDataset"
    subject = "2h5u"
    sess = "postimp"
    datatype = "ieeg"
    suffix = "ieeg"
    run = "03"
    extension = "vhdr"
    
    bids_paths = mne_bids.BIDSPath(
        root=bids_root, 
        session=sess, 
        subject=subject, 
        datatype=datatype, 
        suffix=suffix,
        run=run, 
        extension=extension
    )
    bids_path = bids_paths.match()[0]
    
    # Load raw data
    print("Loading raw data...")
    raw = mne_bids.read_raw_bids(bids_path)
    
    # Create brain visualization
    print("Creating brain visualization...")
    brain = mne.viz.Brain(
        f"sub-{subject}",
        subjects_dir=r"D:\DABI\StimulationDataset\derivatives\freesurfer",
        alpha=0.7, 
        surf="pial",
        show=False
    )
    
    trans = mne.transforms.Transform(fro="head", to="mri", trans=np.eye(4))
    brain.add_sensors(raw.info, trans=trans)
    brain.add_annotation("aparc", borders=False, alpha=0.2)
    
    info = raw.info
    t1 = nb.load(r"D:\DABI\StimulationDataset\sub-4r3o\ses-preimp\anat\sub-4r3o_ses-preimp_acq-T1w_run-01_T1w.nii")
    
    mri_fig, mri_axes = setup_mri_picking(brain, info, t1)
    brain._renderer.plotter.enable_surface_point_picking(False)
    brain.show()

    print("\n" + "="*60)
    print("READY: Click on electrodes to update MRI slices")
    print("Brain surface is now transparent to picking")
    print("Close the brain window when done")
    print("="*60 + "\n")
    
    input("Press Enter to exit...")