import neurokit2 as nk
import numpy as np
import pandas as pd

import mne

from ecg_wrappers import _load_ecg_peaks, _peaks_from_intervals
from annotations_utils import _annotations_start_stop_improved, _onsets_ends_to_intervals, _intervals_intersection

def compute_hrv_time_neurokit(
        raw: mne.io.BaseRaw, 
        ecg_ch_name: str | None = None,
        peaks_ch_name: str | None = None,
        annotation_periods: str | list[str | list[str]] | dict[str: list[str] | str] | None = None,
        annotations_to_keep: str | list[str] | None = None, 
        annotations_to_reject: str | list[str] | None = None, 
        tmin: int | float | None = 0.0,
        tmax: int | float | None = None, 
        min_segment_time: int | float | None = None,
        verbose: bool = True
): 
    sfreq = raw.info["sfreq"]
    peaks = _load_ecg_peaks(
        raw = raw, 
        ch_name = peaks_ch_name, 
    )
    valid_onsets, valid_ends = _annotations_start_stop_improved(
        raw = raw, 
        annotations_to_keep = annotations_to_keep, 
        annotations_to_reject = annotations_to_reject, 
        tmin = tmin, 
        tmax = tmax, 
        min_segment_time = min_segment_time, 
        verbose = verbose
    )
    valid_intervals = _onsets_ends_to_intervals(valid_onsets, valid_ends)
    annotation_periods = _sanitize_annotations_periods(annotation_periods)
    output_dict = {}
    for annotation_name, annotation_period in annotation_periods.items(): 
        # Can technically combine this code with the previous _annotations_start_stop_improved, TO DO
        intervals_annotation_period = _onsets_ends_to_intervals(
            _annotations_start_stop_improved(
                raw, annotations_to_keep = annotation_period
            )
        )
        intervals_annotation_period = _intervals_intersection(
            intervals_annotation_period, valid_intervals
        )
        peaks_period = _peaks_from_intervals(
            intervals = intervals_annotation_period, events = peaks
        )
        # For now, simply use the nk function, and let the function itself determine if the intervals are successive
        out = nk.hrv_time(peaks_period, sampling_rate = sfreq).to_dict()
        output_dict[annotation_name] = out
    return pd.DataFrame.from_dict(
        output_dict, orient = "index"
    )

def _sanitize_annotations_periods(annotation_periods: str | list[str | list[str]] | dict[str: list[str] | str] | None) -> dict[str: list[str]]:
    empty_row_name = ""
    if annotation_periods is None: 
        return {"": None}
    elif isinstance(annotation_periods, str): 
        return {annotation_periods: [annotation_periods]}
    elif isinstance(annotation_periods, list): 
        return {str(annotation_period): ([annotation_period] if isinstance(annotation_period, str) else annotation_period) for annotation_period in annotation_periods}
    elif isinstance(annotation_periods, dict): 
        return {annotation_name: ([annotation_period] if isinstance(annotation_period, str) else annotation_period) for annotation_name, annotation_period in annotation_periods.items()}


if __name__ == "__main__": 
    import mne_bids
    import mne

    from ecg_wrappers import find_ecg_events_neurokit, _add_data_to_raw

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
    raw.load_data()
    #ecg_process_neurokit(raw)
    events, _, _ = find_ecg_events_neurokit(raw, keep_by_annotations = None)
    _add_data_to_raw(raw, events[:, 0], "ecg_peaks")
    print(compute_hrv_time_neurokit(raw, peaks_ch_name = "ecg_peaks"))