import mne
from mne.utils import verbose

import numpy as np
from typing import Callable

from annotations_utils import _annotations_start_stop_improved

@verbose
def find_events(
        raw: mne.io.BaseRaw, 
        pick: str | int | list[str] | list[int], 
        event_finder: Callable,
        event_id: int = 1, 
        tstart: float | int = 0.0, 
        tend: float | int = None,
        min_segment_time: int | float | None = None,
        clean: Callable | None = None, 
        keep_by_annotations: list[str] | str | None = None,
        reject_by_annotations: list[str] | str | None = ["edge", "bad"], 
        annotate_valid_period: str | None = None,
        verbose: bool = True, 
        **kwargs
) -> tuple[np.ndarray | None, int | None, float | None]:
    """ 
    
    """
    #Added sfreq here, as needed for neurokit2.ecg_peaks
    sfreq = raw.info["sfreq"]
    onsets, ends = _annotations_start_stop_improved(
        raw = raw,
        annotations_to_keep = keep_by_annotations, 
        annotations_to_reject = reject_by_annotations,
        tmin = tstart, 
        tmax = tend, 
        min_segment_time = min_segment_time,
        verbose = verbose
    )
    if not len(onsets):
        #Then have found no appropriate segments
        Warning("No segments were appropriate for ECG Peak Extraction")
        return None, None, None
    peaks = [[]]*len(onsets) #Allows for future parallelization if necessary
    for i, (onset, end) in enumerate(zip(onsets, ends)):
        segment, _ = raw[pick, onset:end]
        if clean is not None:
            #Try to avoid transients if possible
            #Further, corresponding ecg_peaks have the same ecg_clean method
            segment = clean(segment, sfreq)
        ecg_segment_peaks = event_finder(segment, sfreq, **kwargs)
        #Need to re-align it with the original onset
        ecg_segment_peaks = ecg_segment_peaks + onset
        peaks[i] = ecg_segment_peaks
    #First eliminate the empty windows - CHECK BACK LATER
    peaks = [peak for peak in peaks if len(peak)]
    peaks_combined = np.concatenate(peaks)
    n_peaks = len(peaks_combined)
    if not n_peaks:
        Warning("No peaks were found")
        return None, None, None
    #Now Annotate the valid ecg periods
    if annotate_valid_period is not None:
        ecg_annotations = mne.Annotations(
            onset = onsets/sfreq,
            duration = (ends - onsets)/sfreq,
            description = annotate_valid_period, 
        )
        #Now add to existing annotations
        raw.set_annotations(raw.annotations + ecg_annotations)
    average_rate = _average_rate_from_windows(peaks, sfreq)
    return (
        np.stack([
            peaks_combined, 
            np.zeros(n_peaks, dtype = int), 
            np.ones(n_peaks, dtype = int)*event_id
        ], axis = 1), 
        pick,
        average_rate
        )


def _average_rate_from_windows(
        peaks: list[list[int]] | list[int],
        sfreq: int
) -> float:
    if not len(peaks):
        return None
    if isinstance(peaks[0], int): 
        peaks = [peaks]
    #First remove all the empty windows - CAN REMOVE LATER
    peaks = [peak_win for peak_win in peaks if len(peak_win)]
    n_times = np.sum([np.ptp(peak_win) for peak_win in peaks])
    n_segs = np.sum([len(peak_win) - 1 for peak_win in peaks])
    if n_segs:
        return (n_segs/n_times)*sfreq
    return None


def sliding_window_accept_reject(
        raw,
        pick: str | int | list[str] | list[int],
        accept_reject_func: Callable, 
        window_time_sec: int | float = 30,
        window_overlap_sec: int | float = 0, #TO FIX, Would probably need to remove this window_overlap_sec parameter
        tstart: int | float | None = 0.0,
        tend: int | float | None = None,
        valid_annotations: str | list[str] | None = None,
        reject_by_annotations: str | list[str] | None = None,
        annotations_name: str | None = "ecg_acceptable", 
        verbose = True,
        **kwargs
): 
    sfreq = raw.info["sfreq"]
    window_N = int(window_time_sec*sfreq)
    window_overlap_N = int(window_overlap_sec*sfreq)
    assert window_overlap_N < window_N
    onsets, ends = _annotations_start_stop_improved(
        raw = raw,
        annotations_to_keep = valid_annotations, 
        annotations_to_reject = reject_by_annotations,
        tmin = tstart, 
        tmax = tend
    )
    onsets_quality, ends_quality = [], []
    for i, (onset, end) in enumerate(zip(onsets, ends)):
        curr_window_acceptable_onset = None
        curr_window_acceptable_end = None
        for window_onset in range(onset, end - window_N, window_N - window_overlap_N):
            window_end = window_onset + window_N
            segment, _ = raw[pick, window_onset:window_end]
            if accept_reject_func(segment, sfreq, **kwargs):
                if curr_window_acceptable_onset is None:
                    curr_window_acceptable_onset = window_onset
                curr_window_acceptable_end = window_end
            else:
                #Quality has dropped, save the current segment if there is one
                if curr_window_acceptable_end is not None:
                    onsets_quality.append(curr_window_acceptable_onset)
                    ends_quality.append(curr_window_acceptable_end)
                    curr_window_acceptable_onset = None
                    curr_window_acceptable_end = None
        #Save the final segment if it exists
        if curr_window_acceptable_end is not None:
            onsets_quality.append(curr_window_acceptable_onset)
            ends_quality.append(curr_window_acceptable_end)
    #now annotate the raw object
    #First convert back to np.ndarray
    onsets_quality = np.array(onsets_quality, dtype = int)
    ends_quality = np.array(ends_quality, dtype = int)
    if not annotations_name is None:
        annotations = mne.Annotations(
            onset = onsets_quality/sfreq,
            duration = (ends_quality - onsets_quality)/sfreq,
            description = annotations_name, 
        )
        raw.set_annotations(raw.annotations + annotations)
    return onsets_quality, ends_quality

