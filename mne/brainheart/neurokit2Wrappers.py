import neurokit2 as nk

import numpy as np

from mne.preprocessing.ecg import _get_ecg_channel_index #Keep it aligned with the original MNE
import mne
from mne.utils import logger, verbose
from mne.annotations import _annotations_starts_stops

def modify_parameters_wrapper(new_params): 
    """_summary_

    Args:
        new_params (_type_): _description_
    """
    def decorator(): 
        pass
    pass

@verbose
def find_ecg_events_neurokit(
        raw: mne.io.BaseRaw, 
        event_id: int = 0, 
        ch_name: str = None, 
        tstart: float | int = 0.0, 
        tend: float | int = None,
        min_segment_time: int | float | None = None,
        method: str = "neurokit", 
        clean: bool = True, 
        reject_by_annotation: list[str] | str | None = ["edge", "bad"], 
        annotate_valid_ecg_period: str | None = "ecg_valid",
        verbose = True
):
    """ Calls ecg_peaks from Neurokit2
    
    """
    #Added sfreq here, as needed for neurokit2.ecg_peaks
    sfreq = raw.info["sfreq"]
    #This section follows from the original preprocessing/find_ecg_events function
    skip_by_annotation = [] if reject_by_annotation is None else reject_by_annotation
    del reject_by_annotation
    idx_ecg = _select_single_ecg_channel(raw, ch_name, return_data=False)
    ecg = raw.get_data(picks = idx_ecg)[0] #IF POSSIBLE DEPRECATE
    onsets, ends = _annotations_starts_stops_time_restriction(
        raw, skip_by_annotation, "reject_by_annotation", invert=True,
        tmin = tstart, tmax = tend, crop_annotations = True, verbose = verbose
    )
    #Further filter by minimum time
    if min_segment_time is not None:
        segments_to_keep = (ends - onsets)/sfreq >= min_segment_time
        onsets, ends = onsets[segments_to_keep], ends[segments_to_keep]
    peaks = [[]]*len(onsets) #Allows for future parallelization if necessary
    for i, (onset, end) in enumerate(zip(onsets, ends)):
        ecg_segment = raw[idx_ecg, onset:end]
        if clean is not None:
            #Try to avoid transients if possible
            #Further, corresponding ecg_peaks have the same ecg_clean method
            ecg_segment = nk.ecg_clean(ecg, sfreq, method = method)
        ecg_segment_peaks = nk.ecg_findpeaks(
            ecg_segment, sfreq, method = method)["ECG_R_Peaks"]
        #Need to re-align it with the original onset
        ecg_segment_peaks = ecg_segment_peaks + onset
        peaks[i] = ecg_segment_peaks
    #First eliminate the empty windows - CHECK BACK LATER
    peaks = [peak for peak in peaks if len(peak)]
    #Keep the average HR to return it later
    average_hr = _average_HR_from_windows(peaks, sfreq)
    peaks_combined = np.concatenate(peaks)
    n_peaks = len(peaks_combined)
    #Now Annotate the valid ecg periods
    if annotate_valid_ecg_period is not None:
        ecg_annotations = mne.Annotations(
            onset = onsets/sfreq,
            duration = (ends - onsets)/sfreq,
            description = annotate_valid_ecg_period, 
        )
        #Now add to existing annotations
        raw.set_annotations(raw.annotations + ecg_annotations)
    
    return (
        np.stack([
            peaks_combined, 
            np.zeros(n_peaks, dtype = int), 
            np.ones(n_peaks, dtype = int)*event_id
        ], axis = 1), 
        idx_ecg,
        average_hr
        )

def ecg_quality_zhao2018_neurokit(raw, ch_name):
    pass

@verbose
def ecg_quality_sliding_window_zhao2018_neurokit(
        raw,
        ch_name: str | None = None, 
        window_time_sec: int | float = 30,
        window_overlap_sec: int | float = 0,
        tstart: int | float | None = 0.0,
        tend: int | float | None = None,
        valid_ecg_annotation: str | list[str] | None = "ecg_valid",
        annotation_name: str | None = "ecg_excellent", 
        keep_barely_acceptable: bool = False,
        verbose = True,
        **kwargs
):  
    sfreq = raw.info["sfreq"]
    window_N = int(window_time_sec*sfreq)
    window_overlap_N = int(window_overlap_sec*sfreq)
    assert window_overlap_N < window_N
    outcomes_to_keep = ["Excellent"]
    if keep_barely_acceptable:
        outcomes_to_keep.append("Barely acceptable")
    idx_ecg, _ = _select_single_ecg_channel(raw, ch_name, return_data=True)
    valid_ecg_annotation = [] if valid_ecg_annotation is None else valid_ecg_annotation
    onsets, ends = _annotations_starts_stops_time_restriction(
        raw, valid_ecg_annotation, "valid_ecg_annotation", invert=False,
        tmin = tstart, tmax = tend, crop_annotations = True, verbose = verbose
    )
    onsets_quality, ends_quality = [], []
    for i, (onset, end) in enumerate(zip(onsets, ends)):
        curr_window_acceptable_onset = None
        curr_window_acceptable_end = None
        for window_onset in range(onset, end - window_N, window_N - window_overlap_N):
            window_end = window_onset + window_N
            ecg_segment = raw[ecg_idx, window_onset:window_end][0][0]
            outcome = nk.ecg_quality(
                ecg_segment, 
                rpeaks = None,
                sampling_rate = sfreq,
                method = "zhao2018", 
                **kwargs
            )
            if outcome in outcomes_to_keep:
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
    onsets_quality = np.array(onsets_quality)
    ends_quality = np.array(ends_quality)
    ecg_annotations = mne.Annotations(
        onset = onsets_quality/sfreq,
        duration = (ends_quality - onsets_quality)/sfreq,
        description = annotation_name, 
    )
    raw.set_annotations(raw.annotations + ecg_annotations)
    return onsets_quality, ends_quality

def ecg_clean_neurokit(
        raw, 
        ch_name = None, 
        method: str = "neurokit", 
        **kwargs
): 
    sfreq = raw.info["sfreq"]
    idx_ecg = _select_single_ecg_channel(
        raw, 
        ch_name, 
        return_data=False
    )

    raw.apply_function(
        _ecg_clean_with_params(sampling_rate=sfreq, method = method, **kwargs),
        picks = idx_ecg, 
        channel_wise = True
    )


def ecg_fixpeaks_neurokit(
        raw, 
        events: np.ndarray,
        event_id: int = 1,
        ch_name: str | None = None,
        event_dict: dict[str:int] = {
            "ectopic": 2,
            "missed": 3,
            "extra": 4,
            "longshort": 5
        },
        iterative: bool = False, #Might replace with **kwargs
        return_peaks_clean: bool = True,
        return_artifacts_dict: bool = False
): 
    ecg_indx = _select_single_ecg_channel(raw, ch_name)
    ecg = raw[ecg_indx, :][0]
    sfreq = raw.info["sfreq"]
    events = events[events[:, 2] == event_id, :]
    artifacts, peaks_clean = nk.signal_fixpeaks(
        peaks = events[:, 0], 
        sampling_rate = sfreq, 
        method = "kubios", 
        iterative = iterative
    )

    for k, v in event_dict.items():
        events[artifacts[k]] = v
    out = (events, event_dict)
    if return_peaks_clean:
        out = out + (peaks_clean)
    if return_artifacts_dict:
        out = out + (artifacts)
    return out
    
    #print(artifacts["drrs"])


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
    annotations_time_restriction_mask =  (Nstart <= ends) & (onsets <= Nend)
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


def _ecg_clean_with_params(
            sampling_rate: int|float, 
            method: str, 
            **kwargs
    ): 
    def inner_func(ecgSignal): 
        return nk.ecg_clean(ecgSignal, sampling_rate = sampling_rate, method = method, **kwargs)
    return inner_func

def _select_single_ecg_channel(raw, ch_name: str = None, return_data = False): 
    idx_ecg = _get_ecg_channel_index(ch_name, raw)
    if idx_ecg is not None:
        logger.info(f"Using channel {raw.ch_names[idx_ecg]} to identify heart beats.")
    else: 
        #The Neurokit2 functions are only tested against real ECG, not simulated
        #As such, we aren't going to apply this function to simulated data
        raise ValueError(
            "No ECG Channel Found"
        )
    if return_data: 
        ecg = raw.get_data(picks = idx_ecg)[0]
        return idx_ecg, ecg
    return idx_ecg


def _average_HR_from_windows(
        peaks: list[list[int]] | list[int],
        sfreq: int
):
    if not len(peaks):
        return None
    if isinstance(peaks[0], int): 
        peaks = [peaks]
    #First remove all the empty windows - CAN REMOVE LATER
    peaks = [peak_win for peak_win in peaks if len(peak_win)]
    n_times = np.sum([np.ptp(peak_win) for peak_win in peaks])
    n_segs = sum(len(peak_win) - 1 for peak_win in peaks)
    if n_segs:
        return (n_segs/n_times)*sfreq*60
    return None



if __name__ == "__main__": 

    import mne_bids
    import mne
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
    #raw.crop(tmax = 60)
    raw.load_data()
    #ecg_clean_neurokit(raw, method = "neurokit")
    print(raw.annotations)
    ecg_clean_neurokit(raw)
    events, ecg_idx, average_hr = find_ecg_events_neurokit(raw)
    print(ecg_quality_sliding_window_zhao2018_neurokit(raw, keep_barely_acceptable=True))