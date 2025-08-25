import neurokit2 as nk

import numpy as np
import pandas as pd

from mne.preprocessing.ecg import _get_ecg_channel_index #Keep it aligned with the original MNE
import mne
from mne.utils import logger, verbose
from mne.annotations import _annotations_starts_stops

from functools import partial

from mne.brainheart.annotations_utils import _annotations_start_stop_improved, _onsets_ends_to_intervals, _intervals_to_onsets_ends
from event_detection import find_events, sliding_window_accept_reject

def modify_parameters_wrapper(new_params): 
    """_summary_

    Args:
        new_params (_type_): _description_
    """
    def decorator(): 
        pass
    pass


def ecg_process_neurokit(
        raw: mne.io.BaseRaw, 
        ecg_ch_name: str | None = None,
        **kwargs
): 
    idx_ecg, ecg_data = _select_single_ecg_channel(raw, ecg_ch_name, return_data = True)
    ecg_dict, ecg_event_indices = nk.ecg_process(ecg_data, sampling_rate = raw.info["sfreq"], **kwargs)
    ch_types = neurokit_ch_names_to_types(ecg_dict.columns)
    _write_events_dict_to_stim(ecg_dict, raw, ch_types = ch_types)
    return raw, ecg_event_indices


def neurokit_ch_names_to_types(
        ch_names: str | list[str],
): 
    if not len(ch_names): 
        return []
    if isinstance(ch_names, str): 
        ch_names = [ch_names]
    elif isinstance(ch_names, pd.Series): 
        ch_names = list(ch_names)
    return [_neurokit_name_to_type(name) for name in ch_names]

def _neurokit_name_to_type(
        name: str
): 
    name = name.lower()
    if "raw" in name or "clean" in name: 
        return "ecg"
    if "rate" in name: 
        return "ecg" # Will need to modify this at some point
    return "stim"

def ecg_quality_reject(
        raw: mne.io.BaseRaw,
        quality_thresh: float = 0.8,
        annotate_quality_name: str = "ecg_quality",
        ecg_quality: np.ndarray | None = None, 
        ecg_quality_ch_name: str | None = "ECG_Quality", 
        min_hr: int | float | None = 40,
        max_hr: int | float | None = 200
): 
    sfreq = raw.info["sfreq"]
    ecg_quality = load_ecg_quality(raw, ecg_quality, ecg_quality_ch_name)
    quality_mask = (quality_thresh <= ecg_quality)
    min_hr = 0 if min_hr is None else min_hr
    max_hr = np.inf if max_hr is None else max_hr
    hr_mask = (min_hr <= ecg_quality) & (ecg_quality <= max_hr)
    final_mask = quality_mask & hr_mask
    intervals_quality = _intervals_from_mask(final_mask)
    onset_quality, ends_quality = _intervals_to_onsets_ends(intervals_quality)
    # Now annotate the raw object
    if not len(intervals_quality):
        ecg_annotations = mne.Annotations(
            onset = onset_quality/sfreq,
            duration = (ends_quality - onset_quality)/sfreq,
            description = annotate_quality_name, 
        )   
        raw.set_annotations(raw.annotations + ecg_annotations)
    else: 
        raise Warning("No Segments With High Enough Quality Found")
    #Now add to existing annotations
    return raw, intervals_quality


def load_ecg_quality(
        raw: mne.io.BaseRaw,
        ecg_quality: np.ndarray | None = None, 
        ecg_quality_ch_name: str | None = None
): 
    if ecg_quality is None: 
        if ecg_quality_ch_name is None: 
            raise ValueError("Please Enter a Value for Either the ECG Quality or the ECG Quality Channel Name")
        ecg_quality = raw.get_data(picks = ecg_quality_ch_name, return_times = False)
    return ecg_quality

@verbose
def find_ecg_events_neurokit( ############# Try and implement A Min and Max HR
        raw: mne.io.BaseRaw, 
        event_id: int = 1, 
        ch_name: str = None, 
        tstart: float | int = 0.0, 
        tend: float | int = None,
        min_segment_time: int | float | None = None,
        method: str = "neurokit", 
        clean: bool = True, 
        keep_by_annotations: list[str] | str | None = "ecg_acceptable",
        reject_by_annotations: list[str] | str | None = ["edge", "bad"], 
        annotate_valid_ecg_period: str | None = "ecg_valid",
        verbose: bool = True
) -> tuple[np.ndarray | None, int | None, float | None]:
    idx_ecg = _select_single_ecg_channel(raw, ch_name, return_data=False)
    nk_ecg_peaks_wrapper = partial(
        lambda ecg_segment, sfreq, method: nk.ecg_peaks(ecg_segment.flatten(), sampling_rate=sfreq, method=method)[1]["ECG_R_Peaks"],
        method=method
    )
    clean_func =  partial(
        lambda ecg_segment, sfreq, method: nk.ecg_clean(ecg_segment.flatten(), sampling_rate=sfreq, method=method),
        method=method
    ) if clean else None
    events, idx_ecg, rate = find_events(
        raw = raw, 
        pick = idx_ecg, 
        event_finder = nk_ecg_peaks_wrapper, 
        event_id = event_id, 
        tstart = tstart, 
        tend = tend, 
        min_segment_time = min_segment_time,
        clean = clean_func, 
        keep_by_annotations = keep_by_annotations,
        reject_by_annotations = reject_by_annotations, 
        annotate_valid_period = annotate_valid_ecg_period,
        verbose = verbose
    )
    return events, idx_ecg, rate*60


@verbose
def ecg_quality_sliding_window_zhao2018_neurokit(
        raw,
        ch_name: str | None = None, 
        window_time_sec: int | float = 30,
        window_overlap_sec: int | float = 0, #TO FIX, Would probably need to remove this window_overlap_sec parameter
        tstart: int | float | None = 0.0,
        tend: int | float | None = None,
        valid_ecg_annotations: str | list[str] | None = None,
        reject_by_annotations: str | list[str] | None = None,
        annotation_name: str | None = "ecg_acceptable", 
        keep_barely_acceptable: bool = False,
        verbose = True,
        **kwargs
):  
    outcomes_to_keep = ["Excellent"]
    if keep_barely_acceptable:
        outcomes_to_keep.append("Barely acceptable")
    ecg_idx, _ = _select_single_ecg_channel(raw, ch_name, return_data=True)
    nk_ecg_quality_wrapper = partial(
        lambda ecg_segment, sfreq, method, **kwargs: (nk.ecg_quality(ecg_segment.flatten(), sampling_rate=sfreq, method=method, **kwargs) in outcomes_to_keep),
        method="zhao2018"
    )
    return sliding_window_accept_reject(
        raw = raw, 
        pick = ecg_idx, 
        accept_reject_func = nk_ecg_quality_wrapper, 
        window_time_sec = window_time_sec, 
        window_overlap_sec = window_overlap_sec, 
        tstart = tstart, 
        tend = tend, 
        valid_annotations = valid_ecg_annotations,
        reject_by_annotations = reject_by_annotations,
        annotations_name = annotation_name, 
        verbose = verbose, 
        **kwargs
    )

def peak_quality_mean_template_neurokit(
        raw: mne.io.BaseRaw, 
        ecg_ch_name: str | None = None,
        peaks_ch_name: str | None = None,
        events: np.ndarray | None = None, 
        event_id: int | None = None,
        annotations_to_keep: str | list[str] | None = ["ecg_valid", "ecg_acceptable"], #Don't know if this is the best way to handle this
        annotations_to_reject: str | list[str] | None = ["bad, edge"],
        method: str = "templatematch", 
): 
    if method.lower() == "zhao2018":
        return ValueError("For Zhao 2018, use ecg_quality_sliding_window_zhao201_neurokit()")
    events = _load_ecg_peaks(raw, peaks_ch_name, events, event_id)
    ecg_idx, ecg_data = _select_single_ecg_channel(raw, ecg_ch_name, return_data = True)
    quality = nk.ecg_quality(
        ecg_data, 
        rpeaks = events[:, 0], 
        sampling_rate = raw.info["sfreq"], 
        method = method
    )
    onsets, ends = _annotations_start_stop_improved(
        raw = raw, 
        annotations_to_keep = annotations_to_keep, 
        annotations_to_reject = annotations_to_reject, 
        min_segment_time = 0
    )
    mask = _mask_from_intervals(_onsets_ends_to_intervals(onsets, ends), raw.n_times)
    return quality*mask


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
    def _ecg_clean_with_params(
            sampling_rate: int|float, 
            method: str, 
            **kwargs
    ): 
        def inner_func(ecgSignal): 
            return nk.ecg_clean(ecgSignal, sampling_rate = sampling_rate, method = method, **kwargs)
        return inner_func

    raw.apply_function(
        _ecg_clean_with_params(sampling_rate=sfreq, method = method, **kwargs),
        picks = idx_ecg, 
        channel_wise = True
    )


def ecg_fixpeaks_neurokit(
        raw, 
        ecg_peaks_ch_name: str | None = "ecg_peaks",
        events: np.ndarray | None = None,
        event_id: int | list[int] = 1,
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
    ecg = raw[ecg_indx, :][0] ############ NEED TO DO ANNOTATION SELECTION
    sfreq = raw.info["sfreq"]
    events = _load_ecg_peaks(
        raw = raw,
        ch_name = ch_name, 
        events = events, 
        event_id = event_id
    )
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
        out = out + (peaks_clean,)
    if return_artifacts_dict:
        out = out + (artifacts,)
    return out


def hr_neurokit2(
        raw: mne.io.BaseRaw, 
        events: np.ndarray | None = None, 
        event_id: int = 1, 
        ch_name: str | None = None, 
        clean_peaks: bool = True, #Will need to implement later
        tmin: int | float | None = 0.0, 
        tmax: int | float | None = None,
        min_segment_time: int | float | None = 10.0,
        annotations_to_keep: str | list[str] | None = None, 
        annotations_to_reject: str | list[str] | None = ["edge", "bad"], 
        annotate_hr: str | None = "hr_valid", 
        min_N_peaks: int = 10, #THIS NUMBER IS ARBITRARY, will probably have to address min N Peaks = 1
        interpolation_method: str = "monotone_cubic"
): 
    sfreq = raw.info["sfreq"]
    events = _load_ecg_peaks(
        raw = raw, 
        ch_name = ch_name, 
        events = events, 
        event_id = event_id
    )
    onsets, ends = _annotations_start_stop_improved(
        raw = raw, 
        annotations_to_keep = annotations_to_keep, 
        annotations_to_reject = annotations_to_reject, 
        tmin = tmin,
        tmax = tmax, 
        min_segment_time = min_segment_time
    )
    intervals = _onsets_ends_to_intervals(onsets, ends)
    peaks = _peaks_from_intervals(intervals, events)
    peaks = _format_peaks(peaks)
    rate_interpolated = np.zeros((1, raw.n_times), dtype = float)
    for peak_win, (onset, end) in zip(peaks, intervals): 
        if len(peak_win) >= min_N_peaks: 
            #Then the window has enough peaks
            win_N = end - onset
            peak_win = peak_win - onset #Align with the window
            rate_win = nk.signal_rate(peak_win, sfreq, desired_length=win_N, interpolation_method = interpolation_method)
            rate_interpolated[0, onset:end] = rate_win    
    return rate_interpolated


def ecg_delineate_neurokit2(
        raw: mne.io.BaseRaw, 
        events: np.ndarray | None = None, 
        event_id: int = 1, 
        ecg_ch_name: str | None = None,
        ecg_peaks_name: str | None = None 
): 
    ecg_idx, ecg_data = _select_single_ecg_channel(raw, ecg_ch_name, True)
    events = _load_ecg_peaks(raw, ecg_peaks_name, events, event_id)
    waves, signals = nk.ecg_delineate(
        ecg_data,
        rpeaks = events[:, 0],
        sampling_rate = raw.info["sfreq"]
    )
    _write_events_dict_to_stim(waves, raw)


# Similar to previous function
def ecg_phase_neurokit2(
        raw: mne.io.BaseRaw, 
        sfreq: int | None = None,
        events: np.ndarray | None = None, 
        event_id: int = 1, 
        ch_name: str | None = None, 
        clean_peaks: bool = True, #Will need to implement later
        tmin: int | float | None = 0.0, 
        tmax: int | float | None = None,
        min_segment_time: int | float | None = 10.0,
        annotations_to_keep: str | list[str] | None = None, 
        annotations_to_reject: str | list[str] | None = ["edge", "bad"], 
        annotate_phase: str | None = "ecg_phase_valid", 
        min_N_peaks: int = 10, #THIS NUMBER IS ARBITRARY, will probably have to address min N Peaks = 1
): 
    sfreq = raw.info["sfreq"] ####### Can Try and Write these to a function
    events = _load_ecg_peaks(
        raw = raw, 
        ch_name = ch_name, 
        events = events, 
        event_id = event_id
    )
    onsets, ends = _annotations_start_stop_improved(
        raw = raw, 
        annotations_to_keep = annotations_to_keep, 
        annotations_to_reject = annotations_to_reject, 
        tmin = tmin,
        tmax = tmax, 
        min_segment_time = min_segment_time
    )
    intervals = _onsets_ends_to_intervals(onsets, ends)
    peaks = _peaks_from_intervals(intervals, events)
    peaks = _format_peaks(peaks)
    ecg_phase = np.zeros((1, raw.n_times), dtype = float)
    for peak_win, (onset, end) in zip(peaks, intervals): 
        if len(peak_win) >= min_N_peaks: 
            #Then the window has enough peaks
            win_N = end - onset
            peak_win = peak_win - onset #Align with the window
            rate_win = nk.ecg_phase(peak_win, sfreq, desired_length=win_N, interpolation_method = interpolation_method)
            ecg_phase[0, onset:end] = rate_win    

    ########## NEED TO REWORK ON THIS
    

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
        return (n_segs/n_times)*sfreq*60
    return None


def _inter_peaks_from_windows(
        peaks: list[list[int]] | list[int],
        sfreq: int
) -> list[list[float]]:
    peaks = _format_peaks(peaks)
    return [np.diff(np.where(peak_win)[0])*sfreq*60 for peak_win in peaks]


def _format_peaks(
        peaks
): 
    if not len(peaks): 
        return [[]]
    if isinstance(peaks[0], int): 
        peaks = [peaks]
    return peaks


def _peaks_from_intervals(intervals, events, event_id: int | None = None):
    if event_id is not None:  
        events = events[events[:, 2] == event_id]
    if not len(intervals): 
        return [[]]
    all_peaks = events[:, 0]
    peaks = [[]]*len(intervals)
    for i, (onset, end) in enumerate(intervals): 
        peaks_mask = (onset <= all_peaks) & (all_peaks <= end)
        peaks[i] = all_peaks[peaks_mask]
    return peaks


def _write_events_to_stim(events: np.ndarray, raw: mne.io.BaseRaw, ch_name: str | None = None): 
    if ch_name is None: 
        return
    data = np.zeros((1, raw.n_times), dtype = int)
    for event_id in np.unique(events[:, 2]): 
        event_mask = events[:, 2] == event_id
        events_index = events[event_mask]
        data[0, events_index] = event_id
    return _write_to_stim(data, raw, ch_name)


def _write_events_dict_to_stim(events_dict: dict, raw: mne.io.BaseRaw, ch_types: list[str] | str | None = None): 
    data = events_dict.values
    if data.ndim == 1: 
        data = np.reshape(data, (len(data), 1))
    data = data.T
    ch_names = events_dict.keys()
    ch_names = list(ch_names)
    if ch_types is None: 
        ch_types = ["stim"]*len(ch_names)
    elif isinstance(ch_types, str): 
        ch_types = [ch_types]*len(ch_names)
    new_info = mne.create_info(ch_names, raw.info["sfreq"], ch_types = ch_types)
    new_raw = mne.io.RawArray(data, new_info)
    return raw.add_channels([new_raw], force_update_info = True)

def _write_to_stim(data: np.ndarray, raw: mne.io.BaseRaw, ch_name: str | None = None): 
    if ch_name is None: 
        return raw
    new_info = mne.create_info([ch_name], raw.info["sfreq"], ch_types = ["stim"])
    if isinstance(data, pd.Series): 
        data = data.values
        data = np.reshape(data, (1, len(data)))
    new_raw = mne.io.RawArray(data, new_info)
    return raw.add_channels([new_raw], force_update_info = True)


def _load_ecg_peaks(raw: mne.io.BaseRaw | None = None, ch_name: str | None = "ecg_peaks", events: np.ndarray | None = None, event_id: int | list[str] | None = None): 
    #Load it from raw
    if events is None: 
        if ch_name is None: 
            ch_name = "ecg_peaks"
        events = mne.find_events(raw, stim_channel = ch_name)
    if event_id is not None: 
        if isinstance(event_id, int):
            events = events[events[:, 2] == event_id]
        else:
            events = events[
                np.any(np.stack(
                    [events[:, 2] == id for id in events], axis = 0
                ), axis = 0)
            ]
    return events


def _mask_from_intervals(intervals, N): 
    mask = np.zeros(N, dtype = bool)
    for onset, end in intervals: 
        mask[onset:end] = True
    return mask


def _intervals_from_mask(mask): 
    # Taken partially from my utils.bool_mask_to_intervals
    if not np.any(mask): 
        return np.array([], int), np.array([], int)
    interval_starts = np.where(mask & np.concatenate([[True], ~mask[:-1]]))[0]
    interval_ends = np.where(mask & np.concatenate([~mask[1:], [True]]))[0]
    return _onsets_ends_to_intervals(interval_starts, interval_ends)
    

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
    raw.load_data()
    #ecg_process_neurokit(raw)
    events = find_ecg_events_neurokit(raw, keep_by_annotations = None)
    print(ecg_quality_sliding_window_zhao2018_neurokit(raw, valid_ecg_annotations = "ecg_valid"))
    '''
    print(_write_events_to_stim(events, raw, "ecg_peaks"))
    rate = hr_neurokit2(raw, events = events, annotations_to_keep="ecg_acceptable")
    import matplotlib.pyplot as plt
    plt.plot(rate[0])

    events_clean, events_dict, peaks_clean = ecg_fixpeaks_neurokit(raw, events)
    N = len(peaks_clean)
    events_clean = np.stack([peaks_clean, np.zeros(N, dtype = int), np.zeros(N, dtype = int)], axis = 1, dtype = int)
    rate_clean = hr_neurokit2(raw, events = events_clean, event_id = None, annotations_to_keep="ecg_acceptable")
    plt.plot(rate_clean[0])
    plt.show()
    '''
    '''
    print(average_hr)
    print(nk.hrv_time(events[:, 0], raw.info["sfreq"]))
    print(ecg_fixpeaks_neurokit(raw, events, 1))
    '''