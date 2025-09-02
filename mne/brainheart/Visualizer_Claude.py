import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mne
import mne_bids
import nibabel as nb

# Set matplotlib to interactive mode and use a GUI backend
matplotlib.use('Qt5Agg')  # or 'TkAgg' if Qt5Agg doesn't work
plt.ion()  # Turn on interactive mode

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
        nearest_pos = None
        for ch in info["chs"]: 
            if ch["kind"] == 802: 
                electrode_pos = ch["loc"][:3]
                distance = np.linalg.norm(click_pos - np.array(electrode_pos))
                if distance < min_distance: 
                    min_distance = distance
                    nearest_electrode = ch["ch_name"]
                    nearest_pos = electrode_pos
        return nearest_electrode, np.array(nearest_pos) * click_pos_adj

    def click_callback(point): 
        print(f"Clicked at 3D position: {point}")
        electrode_name, nearest_pos = find_nearest_electrode(point)
        update_mri_plot(nearest_pos, electrode_name)
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
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Made wider for better viewing
        fig.suptitle('MRI Slices - Click on brain to update', color='white', fontsize=14)
        
        vmin = np.min(data)
        vmax = np.max(data)
        dim = 0.6

        vmean = 0.5 * (vmin + vmax)
        ptp = 0.5 * (vmax - vmin)
        vmax = vmean + (1 + dim) * ptp
        
        # Set titles for each subplot
        axes[0].set_title('Coronal', color='white')
        axes[1].set_title('Sagittal', color='white')  
        axes[2].set_title('Axial', color='white')
        
        # Remove ticks for cleaner look
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        fig.set_facecolor("black")
        
        # Show the figure and keep it open
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to ensure window opens
        
        return fig, axes, vmin, vmax

    def update_mri_plot(point, electrode_name = None): 
        coords = list(point)
        coords_anat = np.array((coords + [1]))
        coords_vox = affine_inv @ coords_anat
        coords_vox_indices = coords_vox.astype(int)[:3]
        
        # Ensure indices are within bounds
        coords_vox_indices[0] = np.clip(coords_vox_indices[0], 0, data.shape[0]-1)
        coords_vox_indices[1] = np.clip(coords_vox_indices[1], 0, data.shape[1]-1)  
        coords_vox_indices[2] = np.clip(coords_vox_indices[2], 0, data.shape[2]-1)

        cmap = "grey"
        subtext = " " + electrode_name if electrode_name is not None else "" 

        # Clear previous images
        for ax in axes:
            ax.clear()

        # Coronal slice
        coronal_slice = data[::-1, :, coords_vox_indices[2]].T
        im0 = axes[0].imshow(coronal_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].axhline(coords_vox_indices[1], color="red", linewidth=2)
        axes[0].axvline(data.shape[0]-coords_vox_indices[0], color="red", linewidth=2)
        axes[0].set_title('Coronal' + subtext, color='white')
        axes[0].text(0.05, 0.95, 'L', transform=axes[0].transAxes, 
                    color='white', fontsize=14, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='left')
        axes[0].text(0.95, 0.95, 'R', transform=axes[0].transAxes, 
                    color='white', fontsize=14, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='right')
        axes[0].text(0.05, 0.05, f"z = {coords_vox_indices[2]}", transform=axes[0].transAxes, 
                    color='white', fontsize=12, fontweight='bold', 
                    verticalalignment='bottom', horizontalalignment='left')

        # Sagittal slice
        sagittal_slice = data[coords_vox_indices[0], :, :]
        im1 = axes[1].imshow(sagittal_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].axhline(coords_vox_indices[1], color="red", linewidth=2)
        axes[1].axvline(coords_vox_indices[2], color="red", linewidth=2)
        axes[1].set_title('Sagittal' + subtext, color='white')
        axes[1].text(0.05, 0.05, f"x = {coords_vox_indices[0]}", transform=axes[1].transAxes, 
                    color='white', fontsize=12, fontweight='bold', 
                    verticalalignment='bottom', horizontalalignment='left')

        # Axial slice  
        horizontal_slice = data[::-1, coords_vox_indices[1], ::-1].T
        im2 = axes[2].imshow(horizontal_slice, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[2].axhline(data.shape[2] - coords_vox_indices[2], color="red", linewidth=2)
        axes[2].axvline(data.shape[0] - coords_vox_indices[0], color="red", linewidth=2)
        axes[2].set_title('Axial' + subtext, color='white')
        axes[2].text(0.05, 0.95, 'L', transform=axes[2].transAxes, 
                    color='white', fontsize=14, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='left')
        axes[2].text(0.95, 0.95, 'R', transform=axes[2].transAxes, 
                    color='white', fontsize=14, fontweight='bold', 
                    verticalalignment='top', horizontalalignment='right')
        axes[2].text(0.05, 0.05, f"y = {coords_vox_indices[1]}", transform=axes[2].transAxes, 
                    color='white', fontsize=12, fontweight='bold', 
                    verticalalignment='bottom', horizontalalignment='left')

        # Remove ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # Add colorbar (remove old one first if it exists)
        if hasattr(fig, '_colorbar'):
            fig._colorbar.remove()
        fig._colorbar = fig.colorbar(im1, ax=axes, shrink=0.8, aspect=20, pad=0.02)
        fig._colorbar.ax.tick_params(colors='white')

        # Update the display
        plt.draw()
        plt.pause(0.01)  # Small pause to ensure update

    # Initialize data
    data = t1.get_fdata()
    affine = t1.affine  # voxel -> Anat
    affine_inv = np.linalg.inv(affine)  # Anat -> Voxel

    # Create the persistent matplotlib figure
    fig, axes, vmin, vmax = create_mri_plot()
    
    # Initialize with center point
    center_point = (0, 0, 0)
    update_mri_plot(center_point)
    
    # Setup the brain interaction
    plotter = brain._renderer.plotter
    add_name_labels(info, plotter)
    plotter.track_click_position(callback=click_callback)
    plotter.enable_trackball_style()

    # Return figure reference to keep it alive
    return fig, axes


if __name__ == "__main__": 
    import mne_bids
    import nibabel as nb
    
    bids_root = r"D:/DABI/StimulationDataset"
    ext = "vhdr"
    subject = "2h5u"
    sess = "postimp"
    datatype = "ieeg"
    suffix = "ieeg"
    run = "03"
    extension = "vhdr"
    
    bids_paths = mne_bids.BIDSPath(root=bids_root, 
                                session=sess, 
                                subject=subject, 
                                datatype=datatype, 
                                suffix=suffix,
                                run=run, 
                                extension=extension)
    bids_path = bids_paths.match()[0]
    
    # Load data
    raw = mne_bids.read_raw_bids(bids_path)

    # Set 3D backend
    mne.viz.set_3d_backend('pyvista')

    # Create brain
    brain = mne.viz.Brain(
        f"sub-{subject}",
        subjects_dir=r"D:\DABI\StimulationDataset\derivatives\freesurfer",
        alpha=0.7, 
        show=False)
    
    trans = mne.transforms.Transform(fro="head", to="mri", trans=np.eye(4))
    brain.add_sensors(raw.info, trans=trans)
    brain.add_annotation("aparc", borders=False, alpha=0.2)
    
    info = raw.info
    t1 = nb.load(rf"D:\DABI\StimulationDataset\sub-{subject}\ses-preimp\anat\sub-{subject}_ses-preimp_acq-T1w_run-01_T1w.nii")
    
    # Setup MRI picking and keep figure reference
    mri_fig, mri_axes = setup_mri_picking(brain, info, t1)
    
    # Show brain (this should be last)
    brain.show()
    
    # Keep the script running
    print("Click on the brain to update MRI slices. Close the brain window to exit.")
    input("Press Enter to exit...")  # Keeps script alive