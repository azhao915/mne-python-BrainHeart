import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np

from config_colors import fs_colors

def create_mri_plot(data): 
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

    return fig, axes, vmin, vmax


def update_mri_plot(point, affine_inv, axes, data, fig, vmin, vmax, electrode_name = None): 
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

    if electrode_name: 
        title_color = "red"
        title_suffix = f" - {electrode_name}"
    else: 
        title_color = "white"
        title_suffix = f" - No electrode selected"

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
    axes[1].set_title(f"Sagittal {title_suffix}", color = title_color)

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
    print(1)
    print(2)
    plt.draw()
    plt.pause(0.01)

    return coronal_slice, sagittal_slice, horizontal_slice

def label_mri(point, affine_inv, subcortical_labels, axes): 
    coords = list(point)
    coords_anat = np.array((coords + [1]))
    coords_vox = affine_inv @ coords_anat
    coords_vox_indices = coords_vox.astype(int)[:3]

    for coord_index in range(3): 
        coords_vox_indices[coord_index] = np.clip(coords_vox_indices[coord_index], 0, subcortical_labels.shape[coord_index]-1)

    coronal_labels = subcortical_labels[::-1, :, coords_vox_indices[2]].T
    sagittal_labels = subcortical_labels[coords_vox_indices[0], :, :]
    horizontal_labels = subcortical_labels[::-1, coords_vox_indices[1], ::-1].T

    unique_labels = np.sort(np.unique(np.stack(coronal_labels, sagittal_labels, horizontal_labels)))

    colors = [fs_colors[label] if label in fs_colors.keys() else (0.5, 0.5, 0.5) for label in unique_labels]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(unique_labels) + [np.max(unique_labels) + 1], len(colors))

    for data, ax in zip([coronal_labels, sagittal_labels, horizontal_labels], axes):
        ax.imshow(data, origin = "lower", alpha = 0.3, cmap = cmap, norm = norm, interpolation = "nearest")

