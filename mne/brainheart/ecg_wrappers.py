import neurokit2 as nk

import numpy as np
import pandas as pd

from mne.preprocessing.ecg import _get_ecg_channel_index #Keep it aligned with the original MNE
import mne
from mne.utils import logger, verbose

from functools import partial, wraps

from mne.brainheart.annotations_utils import (
    _annotations_start_stop_improved, 
    _onsets_ends_to_intervals, 
    _intervals_to_onsets_ends, 
    _intervals_intersection,
    _intervals_subtraction_boolean, 
    write_to_annotations)

from mne.brainheart.utils import (
    _intervals_from_mask, 
    _write_events_dict_to_stim, 
    _peaks_from_intervals, 
    _format_peaks, 
    _mask_from_intervals, 
    _add_data_to_raw, 
    write_events_to_channel)


from mne.brainheart.loading.ecg_loading import (
    load_ecg, 
    load_ecg_peaks, 
    _select_single_ecg_channel, 
    load_hr, 
    load_ecg_peaks_corrected
)


from mne.brainheart.event_detection import (
    find_events, 
    sliding_window_accept_reject)

from mne.brainheart.ecg_channel_names_enum import ECG_Channels
from mne.brainheart.ecg_annotations_enum import ECG_Annotations

def ecg_process_neurokit(
        raw: mne.io.BaseRaw, 
        ecg_ch_name: str | None = None,
        **kwargs
): 
    _, ecg_data = _select_single_ecg_channel(raw, ecg_ch_name, return_data = True)
    ecg_data = ecg_data.flatten()
    ecg_df, ecg_event_indices = nk.ecg_process(ecg_data, sampling_rate = raw.info["sfreq"], **kwargs)

    # Now convert the ecg_event_indices to a dict for the relevant measures
    ecg_df = pd.concat([ecg_df, _ecg_event_indices_from_preprocess_to_dict(ecg_event_indices, raw.n_times)], axis = 1)
    ch_types = neurokit_ch_names_to_types(ecg_df.columns)
    _write_events_dict_to_stim(ecg_df, raw, ch_types = ch_types)
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
    if "peaks" in name or "fixpeaks" in name: 
        return "stim"
    return "ecg"


def _ecg_event_indices_from_preprocess_to_dict(
        ecg_event_indices,
        n_times, 
        keys_to_columns = ["ECG_R_Peaks", 
                           "ECG_fixpeaks_ectopic", 
                           "ECG_fixpeaks_extra", 
                           "ECG_fixpeaks_longshort"], 
        column_names = [ECG_Channels.ECG_R_Peaks_Corrected.value, 
                        ECG_Channels.ECG_fixpeaks_ectopic.value, 
                        ECG_Channels.ECG_fixpeaks_extra.value, 
                        ECG_Channels.ECG_fixpeaks_longshort.value]) -> pd.DataFrame: 
    events_all = np.zeros((n_times, len(column_names)), dtype = int)
    for i, key in enumerate(keys_to_columns):
        events_all[ecg_event_indices[key], i] = 1
    return pd.DataFrame(data = events_all, columns=column_names)



def ecg_quality_reject(
        raw: mne.io.BaseRaw,
        events: np.ndarray | None = None,
        event_id: int | None = None,
        quality_thresh: float = 0.8,
        annotate_quality_name: str = "ECG_Quality",
        quality: np.ndarray | None = None, 
        quality_ch_name: str | None = ECG_Channels.ECG_Quality.value, 
        min_hr: int | float | None = 40,
        max_hr: int | float | None = 200, 
        hr: np.ndarray | None = None
): 
    #Also written to work with PPG
    sfreq = raw.info["sfreq"]
    ecg_quality = load_ecg_quality(raw, quality, quality_ch_name)
    quality_mask = (quality_thresh <= ecg_quality)
    if min_hr is not None and max_hr is not None:
        hr_mask = _hr_mask(
            raw = raw, 
            hr = hr, 
            events = events, 
            event_id = event_id, 
            min_hr = min_hr, 
            max_hr = max_hr 
        )
        quality_mask = quality_mask & hr_mask
    intervals_quality = _intervals_from_mask(quality_mask)
    onsets_quality, ends_quality = _intervals_to_onsets_ends(intervals_quality)
    # Now annotate the raw object
    if not len(intervals_quality):
        write_to_annotations(
            raw,
            onsets_quality, 
            ends_quality, 
            desc = annotate_quality_name)
    else: 
        print("No Segments With High Enough Quality Found")
    #Now add to existing annotations
    return raw, intervals_quality


@verbose
def find_ecg_events_neurokit( ############# Try and implement A Min and Max HR
        raw: mne.io.BaseRaw, 
        event_id: int = 1, 
        ch_name: str | None = None, 
        tstart: float | int = 0.0, 
        tend: float | int | None = None,
        min_segment_time: int | float | None = None,
        method: str = "neurokit", 
        clean: bool = True, 
        ecg_peak_ch_name = ECG_Channels.ECG_R_Peaks.value,
        keep_by_annotations: list[str] | str | None = ECG_Annotations.ECG_Valid.value,
        reject_by_annotations: list[str] | str | None = ["edge", "bad"], 
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
        verbose = verbose
    )
    write_events_to_channel(events, raw, ecg_peak_ch_name, "ecg")
    if rate is None: 
        return events, idx_ecg, rate
    return events, idx_ecg, rate*60


def ecg_quality_sliding_window_zhao2018_neurokit(
        raw,
        ch_name: str | None = None, 
        window_time_sec: int | float = 30,
        window_overlap_sec: int | float = 0,
        tstart: int | float | None = 0.0,
        tend: int | float | None = None,
        valid_ecg_annotations: str | list[str] | None = ECG_Annotations.ECG_Valid.value,
        reject_by_annotations: str | list[str] | None = None,
        annotation_reject_name: str | None = ECG_Annotations.bad_ECG_ZHAO2018.value, 
        keep_barely_acceptable: bool = False,
        verbose = True,
        **kwargs
):  
    outcomes_to_keep = ["Excellent"]
    if keep_barely_acceptable:
        outcomes_to_keep.append("Barely acceptable")
    ecg_idx = _select_single_ecg_channel(raw, ch_name, return_data= False)
    nk_ecg_quality_wrapper = partial(
        lambda ecg_segment, sfreq, method, **kwargs: (nk.ecg_quality(ecg_segment.flatten(), sampling_rate=sfreq, method=method, **kwargs) in outcomes_to_keep),
        method="zhao2018"
    )
    onsets_valid, ends_valid = _annotations_start_stop_improved(
        raw = raw, 
        annotations_to_keep = valid_ecg_annotations, 
        annotations_to_reject = reject_by_annotations, 
        tmin = tstart, 
        tmax = tend, 
        min_segment_time = window_time_sec
    )
    onsets_to_keep, ends_to_keep = sliding_window_accept_reject(
        raw = raw, 
        pick = ecg_idx, 
        accept_reject_func = nk_ecg_quality_wrapper, 
        window_time_sec = window_time_sec, 
        window_overlap_sec = window_overlap_sec, 
        onsets = onsets_valid, 
        ends = ends_valid,
        verbose = verbose, 
        **kwargs
    )
    # Now invert
    intervals_reject = _intervals_subtraction_boolean(
        _onsets_ends_to_intervals(onsets_valid, ends_valid), 
        _onsets_ends_to_intervals(onsets_to_keep, ends_to_keep)
    )
    onsets_reject, ends_reject = _intervals_to_onsets_ends(intervals_reject)
    write_to_annotations(
        raw = raw, 
        onsets = onsets_reject, 
        ends = ends_reject, 
        desc = annotation_reject_name
    )
    return onsets_reject, ends_reject


def peak_quality_mean_template_neurokit(
        raw: mne.io.BaseRaw, 
        ecg_ch_name: str | None = None,
        peaks_ch_name: str | None = None,
        events: np.ndarray | None = None, 
        event_id: int | None = None,
        annotations_to_keep: str | list[str] | None = ECG_Annotations.ECG_Valid.value,
        annotations_to_reject: str | list[str] | None = ["bad"],
        method: str = "templatematch", 
): 
    if method.lower() == "zhao2018":
        return ValueError("For Zhao 2018, use ecg_quality_sliding_window_zhao201_neurokit()")
    events = load_ecg_peaks(raw, peaks_ch_name, events, event_id)
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
        raw_ch_name: str = ECG_Channels.ECG_Raw.value,
        clean_ch_name: str = ECG_Channels.ECG_Clean.value, 
        method: str = "neurokit", 
        **kwargs
): 
    sfreq = raw.info["sfreq"]
    idx_ecg = _select_single_ecg_channel(
        raw, 
        raw_ch_name, 
        return_data=False
    )
    def _ecg_clean_with_params(
            sampling_rate: int|float, 
            method: str, 
            **kwargs
    ): 
        def inner_func(ecgSignal): 
            return nk.ecg_clean(ecgSignal.flatten(), sampling_rate = sampling_rate, method = method, **kwargs)
        return inner_func

    raw.apply_function(
        _ecg_clean_with_params(sampling_rate=sfreq, method = method, **kwargs),
        picks = idx_ecg, 
        channel_wise = True
    )
    mne.rename_channels(
        raw.info, 
        mapping = {
            raw_ch_name: clean_ch_name
        })


def ecg_fixpeaks_neurokit(
        raw, 
        ecg_peaks_ch_name: str | None = ECG_Channels.ECG_R_Peaks.value,
        events: np.ndarray | None = None,
        event_id: int | list[int] = 1,
        ecg_ch_name: str | None = None,
        ecg_peaks_corrected_ch_name: str | None = ECG_Channels.ECG_R_Peaks_Corrected.value,
        event_dict: dict[str:int] = {
            "ectopic": 2,
            "missed": 3,
            "extra": 4,
            "longshort": 5
        },
        iterative: bool = False,
        return_peaks_clean: bool = True,
        return_artifacts_dict: bool = False, 
        artifacts_ch_names: dict | None = {
            "ectopic": ECG_Channels.ECG_fixpeaks_ectopic.value, 
            "missed": ECG_Channels.ECG_fixpeaks_missed.value, 
            "extra": ECG_Channels.ECG_fixpeaks_extra.value, 
            "longshort": ECG_Channels.ECG_fixpeaks_longshort.value
        },
): 
    sfreq = raw.info["sfreq"]
    events = load_ecg_peaks(
        raw = raw,
        peak_ch_name = ecg_peaks_ch_name, 
        events = events, 
        event_id = event_id
    )
    artifacts, peaks_clean = nk.signal_fixpeaks(
        peaks = events[:, 0], 
        sampling_rate = sfreq, 
        method = "kubios", 
        iterative = iterative
    )
    write_events_to_channel(
        events = peaks_clean, 
        raw = raw, 
        ch_name = ecg_peaks_corrected_ch_name, 
        ch_type = "ecg"
    )
    for k, v in event_dict.items():
        events[artifacts[k]] = v
        if artifacts_ch_names is not None: 
            write_events_to_channel(
                events = np.array(artifacts[k]), 
                raw = raw, 
                ch_name = artifacts_ch_names[k], 
                ch_type = "ecg"
            )
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
        peak_ch_name: str | None = None, 
        tmin: int | float | None = 0.0, 
        tmax: int | float | None = None,
        min_segment_time: int | float | None = 10.0,
        annotations_to_keep: str | list[str] | None = ECG_Annotations.ECG_Valid.value, 
        annotations_to_reject: str | list[str] | None = ["edge", "bad"], 
        hr_ch_name: str | None = ECG_Channels.ECG_Rate.value,
        min_N_peaks: int = 10, #THIS NUMBER IS ARBITRARY, will probably have to address min N Peaks = 1
        interpolation_method: str = "monotone_cubic"
): 
    sfreq = raw.info["sfreq"]
    events = load_ecg_peaks(
        raw = raw, 
        peak_ch_name = peak_ch_name, 
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
    if hr_ch_name is not None: 
        _add_data_to_raw(raw, rate_interpolated, hr_ch_name, "ecg")
    return rate_interpolated

@wraps(hr_neurokit2)
def hr_corrected_neurokit(
        raw: mne.io.BaseRaw,
        peak_ch_name = ECG_Channels.ECG_R_Peaks_Corrected.value,
        hr_ch_name = ECG_Channels.ECG_Rate_Corrected.value, 
        **kwargs
): 
    events = load_ecg_peaks_corrected(
        raw = raw, 
        peak_corrected_ch_name = peak_ch_name, 
    )
    return hr_neurokit2(
        raw = raw,
        events = events,
        hr_ch_name = hr_ch_name,
        **kwargs
    )


def ecg_delineate_neurokit2(
        raw: mne.io.BaseRaw, 
        events: np.ndarray | None = None, 
        event_id: int = 1, 
        ecg_ch_name: str | None = None,
        ecg_peaks_name: str | None = None, 
): 
    _, ecg_data = _select_single_ecg_channel(raw, ecg_ch_name, True)
    events = load_ecg_peaks(
        raw = raw, 
        peak_ch_name = ecg_peaks_name, 
        ecg_ch_name = ecg_ch_name, 
        events = events, 
        event_id = event_id)
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
        tmin: int | float | None = 0.0, 
        tmax: int | float | None = None,
        min_segment_time: int | float | None = 10.0,
        annotations_to_keep: str | list[str] | None = ECG_Annotations.ECG_Valid.value, 
        annotations_to_reject: str | list[str] | None = ["edge", "bad"], 
        min_N_peaks: int = 10, #THIS NUMBER IS ARBITRARY, will probably have to address min N Peaks = 1
): 
    sfreq = raw.info["sfreq"] ####### Can Try and Write these to a function
    events = load_ecg_peaks(
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
    ecg_phase_atrial = np.zeros((1, raw.n_times), dtype = float)
    ecg_phase_ventricular = np.zeros((1, raw.n_times), dtype = float)
    for peak_win, (onset, end) in zip(peaks, intervals): 
        if len(peak_win) >= min_N_peaks: 
            #Then the window has enough peaks
            win_N = end - onset
            peak_win = peak_win - onset #Align with the window
            ecg_phase_dict = nk.ecg_phase(peak_win, sfreq, desired_length=win_N, interpolation_method = interpolation_method)
            ecg_phase_atrial[0, onset:end] = ecg_phase_dict["ECG_Phase_Atrial"] + 1
            ecg_phase_ventricular[0, onset:end] = ecg_phase_dict["ECG_Phase_Atrial"] + 1 # We are adding one, as we are starting with zeros, so zeros must represent invalid periods

    return ecg_phase_atrial, ecg_phase_ventricular


def _hr_annotations(
        raw: mne.io.BaseRaw, 
        annotation_name: str | None = None,
        hr: np.ndarray | None = None, 
        events: np.ndarray | None = None, 
        event_id: int | None = None, 
        min_hr: int | float | None = None, 
        max_hr: int | float | None = None, 
        tmin: int | float | None = 0.0, 
        tmax: int | float | None = None,
        min_segment_time: int | float | None = None,
        annotations_to_keep: str | list[str] | None = ECG_Annotations.ECG_Valid.value, #Don't know if this is the best way to handle this
        annotations_to_reject: str | list[str] | None = ["bad, edge"],
): 
    hr_mask = _hr_mask(
        raw = raw, 
        hr = hr, 
        events = events, 
        event_id = event_id, 
        min_hr = min_hr, 
        max_hr = max_hr 
    )
    hr_intervals = _intervals_from_mask(hr_mask)
    onsets, ends = _annotations_start_stop_improved(
        raw = raw, 
        annotations_to_keep = annotations_to_keep, 
        annotations_to_reject = annotations_to_reject, 
        tmin = tmin,
        tmax = tmax, 
        min_segment_time = min_segment_time
    )
    intervals_annotations = _onsets_ends_to_intervals(onsets, ends)
    intervals_combined = _intervals_intersection(hr_intervals, intervals_annotations)
    onsets_combined, ends_combined = _intervals_to_onsets_ends(intervals_combined)
    write_to_annotations(raw, onsets_combined, ends_combined, desc = annotation_name)
    return onsets_combined, ends_combined


def _hr_mask(
        raw: mne.io.BaseRaw, 
        hr: np.ndarray | None = None, 
        events: np.ndarray | None = None, 
        event_id: int | None = None, 
        min_hr: int | float | None = None, 
        max_hr: int | float | None = None
): 
    min_hr = 0 if min_hr is None else min_hr
    max_hr = np.inf if max_hr is None else max_hr
    hr = load_hr(
        raw, 
        hr, 
        events, 
        event_id).flatten()
    hr_mask = (min_hr <= hr) & (hr <= max_hr)
    return hr_mask

def annotate_tachycardia(
        raw: mne.io.BaseRaw, 
        hr: np.ndarray | None = None, 
        events: np.ndarray | None = None, 
        event_id: int | None = None, 
        tmin: int | float | None = 0.0, 
        tmax: int | float | None = None,
        min_segment_time: int | float | None = None,
        annotations_to_keep: str | list[str] | None = ECG_Annotations.ECG_Valid.value,
        annotations_to_reject: str | list[str] | None = ["bad, edge"],
): 
    return _hr_annotations(
        raw = raw, 
        annotation_name = "tachycardia", 
        hr = hr, 
        events = events, 
        event_id = event_id, 
        min_hr = 100, 
        tmin = tmin, 
        tmax = tmax, 
        min_segment_time = min_segment_time,
        annotations_to_keep = annotations_to_keep, 
        annotations_to_reject = annotations_to_reject, 
    )    

def annotate_bradycardia(
        raw: mne.io.BaseRaw, 
        hr: np.ndarray | None = None, 
        events: np.ndarray | None = None, 
        event_id: int | None = None, 
        tmin: int | float | None = 0.0, 
        tmax: int | float | None = None,
        min_segment_time: int | float | None = None,
        annotations_to_keep: str | list[str] | None = ECG_Annotations.ECG_Valid.value,
        annotations_to_reject: str | list[str] | None = ["bad, edge"],
): 
    return _hr_annotations(
        raw = raw, 
        annotation_name = "bradycardia", 
        hr = hr, 
        events = events, 
        event_id = event_id, 
        max_hr = 60, 
        tmin = tmin, 
        tmax = tmax, 
        min_segment_time = min_segment_time,
        annotations_to_keep = annotations_to_keep, 
        annotations_to_reject = annotations_to_reject, 
    )    


if __name__ == "__main__": 
    from load_reference_dataset import load
    raw = load(0)

    from loading.ecg_loading import annotate_valid_ecg_periods, load_ecg, identify_ecg_channel
    annotate_valid_ecg_periods(raw)
    identify_ecg_channel(raw)
    ecg = load_ecg(raw)
    find_ecg_events_neurokit(raw)
    ecg_quality_sliding_window_zhao2018_neurokit(
        raw,
        window_overlap_sec = 15, 
        keep_barely_acceptable=False)
    ecg_clean_neurokit(raw)
    #ecg_fixpeaks_neurokit(raw)
    rate = hr_neurokit2(raw, min_N_peaks = 1)

    ecg_delineate_neurokit2(raw)

    annotate_bradycardia(raw)

    ecg_fixpeaks_neurokit(raw)

    hr_corrected_neurokit(raw)