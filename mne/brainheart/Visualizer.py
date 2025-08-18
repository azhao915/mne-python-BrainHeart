import numpy as np
import mne

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

if __name__ == "__main__": 
    
    import mne_bids
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

    brain = mne.viz.Brain(
        f"sub-{subject}",
        subjects_dir=r"D:\DABI\StimulationDataset\derivatives\freesurfer",
        alpha = 0.7, 
        show = False)
    trans = mne.transforms.Transform(fro="head", to="mri", trans=np.eye(4))
    brain.add_sensors(raw.info, trans=trans)
    brain.add_annotation("aparc", borders = False, alpha = 0.2)
    info = raw.info
    setup_simple_electrode_picking(brain, info)
    brain.show()
