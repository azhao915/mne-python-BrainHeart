import numpy as np

from mne.utils import logger, verbose, _validate_type
from mne.annotations import _annotations_starts_stops, _sync_onset
from mne.io import BaseRaw

@verbose
def _annotations_starts_stops_time_restriction(
        raw, #Probably better to include this function in annotations.py 
        kinds, 
        name, 
        invert = False,
        tmin = 0.0,
        tmax = None,
        crop_annotations: bool = False, 
        verbose: bool = True): 
    onsets, ends = _annotations_starts_stops(
        raw, 
        kinds, 
        name, 
        invert 
    )
    logger.info(f"Found Onsets: {onsets}, Ends: {ends}")
    logger.info(f"Now choosing annotations from [{tmin} to {"end" if tmax is None else tmax}] sec")
    Nstart = 0 if tmin is None else raw.time_as_index(tmin)
    Nend = raw.n_times if tmax is None else raw.time_as_index(tmax)
    annotations_time_restriction_mask =  (Nstart < ends) & (onsets < Nend) #Probably better with <=, >= but then would need to deal with Nstart == Nend
    onsets, ends = onsets[annotations_time_restriction_mask], ends[annotations_time_restriction_mask]
    logger.info(f"Rejected {np.sum(~annotations_time_restriction_mask)} annotations for being completely out of the range")
    del annotations_time_restriction_mask
    tstart_in_annotations = (onsets < Nstart)
    tend_in_annotations = (Nend < ends)
    strict_mask = tstart_in_annotations | tend_in_annotations 
    if crop_annotations:
        onsets[tstart_in_annotations] = Nstart
        ends[tend_in_annotations] = Nend
        logger.info(f"Cropped {np.sum(strict_mask)} annotations")
    else:
        #Then delete the segments where tstart or tend appear in the annotation
        onsets = onsets[~strict_mask]
        ends = ends[~strict_mask]
        logger.info(f"Further Removed {np.sum(strict_mask)} annotations")
    return onsets, ends


@verbose
def _annotations_start_stop_improved(
    raw: BaseRaw,
    annotations_to_keep: str | list[str] | None, 
    annotations_to_reject: str | list[str] | None, 
    combine_annotations_to_keep: bool = True, 
    name: str = "Annotation"
) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        raw (BaseRaw): _description_
        annotations_to_keep (str | list[str] | None): _description_
        annotations_to_reject (str | list[str] | None): _description_
        combine_annotations_to_keep (bool, optional): _description_. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    if len(raw.annotations):
        if annotations_to_keep is None:
            onsets_to_keep = np.array([_sync_onset(raw, raw.annotations._first_time)])
            onsets_to_keep = raw.time_as_index(onsets_to_keep)
            ends_to_keep = onsets_to_keep + raw.n_times
        else: 
            annotations_to_keep = _format_annotation_types(annotations_to_keep)
        if annotations_to_reject is None:
            onsets_to_keep = np.array([], dtype = int)
            ends_to_keep = np.array([], dtype = int)
        else:
            annotations_to_reject = _format_annotation_types(annotations_to_reject)
        onsets_to_keep, ends_to_keep = _onsets_ends_nonoverlapping_from_raw(raw, annotations_to_keep)
        onsets_to_reject, ends_to_reject = _onsets_ends_nonoverlapping_from_raw(raw, annotations_to_reject)
    else: 
        onsets_to_keep = np.array([], int)
        ends_to_keep = np.array([], int)
        onsets_to_reject = np.array([], int)
        ends_to_reject = np.array([], int)
    
    


def _format_annotation_types(annotations): 
    #Takes directly from the annotations.py _annotations_starts_stops function
    _validate_type(annotations, (str, list, tuple))
    if isinstance(annotations, str): 
        annotations = [annotations]
    else:
        for annot in annotations: 
            _validate_type(annot, "str", "All entries")
    return annotations

def _onsets_ends_from_indices(raw, indices): 
    '''Simply fetches them, doesn't combined them non-overlapping segments'''
    #From the annotations.py _annotations_starts_stops function
    onsets = raw.annotations.onset[indices]
    onsets = _sync_onset(raw, onsets)
    ends = onsets + raw.annotations.duration[indices]
    onsets = raw.time_as_index(onsets, use_rounding=True)
    ends = raw.time_as_index(ends, use_rounding=True)
    return onsets, ends


def _onset_ends_nonoverlapping(
        onsets, #sorted already from the annotations object
        ends
): 
    onsets_nonoverlapping, ends_nonoverlapping = [], []
    win_idx = 0
    if not len(onsets):
        return np.ndarray([], dtype = int), np.ndarray([], dtype = int)
    curr_seg_start = onsets[0]
    while win_idx < len(onsets):
        win_onset = onsets[win_idx]
        win_end = ends[win_idx]
        window_mask_to_fuse = (onsets >= win_onset) & (onsets <= win_end) & (win_end < ends)
        if not np.any(window_mask_to_fuse):
            onsets_nonoverlapping.append(curr_seg_start)
            ends_nonoverlapping.append(win_end)
            #Next segment
            mask_for_next_seg = win_end < onsets
            if not np.any(mask_for_next_seg): 
                break
            win_idx = np.where(mask_for_next_seg)[0][0]
            if win_idx < len(onsets):
                curr_seg_start = onsets[win_idx]
        else:
            win_idx = np.argmax(ends*window_mask_to_fuse)
    onsets_nonoverlapping = np.array(onsets_nonoverlapping, dtype = int)
    ends_nonoverlapping = np.array(ends_nonoverlapping, dtype = int)
    return onsets_nonoverlapping, ends_nonoverlapping


def _onsets_ends_nonoverlapping_from_raw(
        raw, 
        annotations: list[str]
): 
    annotations_df = raw.annotations.to_data_frame()
    annotations_desc = annotations_df.description.str.lower()
    mask_to_keep = annotations_desc.startswith(annotations)
    idx_to_keep = np.where(mask_to_keep)[0]
    onsets, ends = _onsets_ends_from_indices(raw, idx_to_keep)
    onsets, ends = _onset_ends_nonoverlapping(onsets, ends)
    return onsets, ends

if __name__ == "__main__":
    onsets = np.array([0, 1, 4, 7])
    ends = np.array([5, 3, 6, 12])
    print(_onset_ends_nonoverlapping(onsets, ends))


def _interval_difference(
        onsets_to_keep,
        ends_to_keep,
        onsets_to_reject,
        ends_to_reject
): 
    final_onsets = []
    final_ends = []
    for onset_to_keep, end_to_keep in zip(onset_to_keep, ends_to_keep): 
        onsets_to_reject_in_seg = (onset_to_keep <= onsets_to_reject) & (onsets_to_reject <= end_to_keep)
        ends_to_reject_in_seg = (onset_to_keep <= ends_to_reject) & (ends_to_reject <= end_to_keep) 
        