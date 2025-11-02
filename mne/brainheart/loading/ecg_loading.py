import mne
import numpy as np

from mne.preprocessing.ecg import _get_ecg_channel_index, _make_ecg

from mne.brainheart.annotations_utils import _annotations_start_stop_improved, write_to_annotations

from mne.brainheart.ecg_annotations_enum import ECG_Annotations
from mne.brainheart.utils import _add_data_to_raw

from mne.brainheart.ecg_channel_names_enum import ECG_Channels

from functools import wraps

def annotate_valid_ecg_periods(
        raw: mne.io.BaseRaw, 
        onsets: np.ndarray | None = None, 
        ends: np.ndarray | None = None, 
        annotation_name: str = ECG_Annotations.ECG_Valid.value
): 
    write_to_annotations(raw, onsets, ends, desc = annotation_name)


def identify_ecg_channel(
        raw: mne.io.BaseRaw, 
        ch_name: str | None = None, 
        ecg_ch_name: str | None = ECG_Channels.ECG_Raw.value
    ) -> int: 
    ecg_chans = mne.channel_indices_by_type(raw.info)["ecg"]
    if not len(ecg_chans): 
        raise ValueError("There are no Found ECG Channels")
    if len(ecg_chans) > 1: 
        raise ValueError("There is more than 1 ECG Channel")
    if ch_name is None:
        ecg_chan_index = ecg_chans[0]
        ch_name = raw.ch_names[ecg_chan_index]
    _validate_ch_name_and_type(raw, ch_name, "ecg")
    if not ch_name == ecg_ch_name: 
        mne.rename_channels(raw.info, {ch_name: ecg_ch_name}, allow_duplicates = False)
        print(f"Renamed {ch_name} to {ecg_ch_name}")
    return ecg_chan_index


def make_synthetic_ecg(
        raw: mne.io.BaseRaw, 
        tmin: int | float | None = 0.0, 
        tmax: int | float | None = None, 
        ecg_ch_name: str | None = ECG_Channels.ECG_Raw.value, 
        reject_by_annotation: bool = False, 
        annotate_valid_periods: str | None = ECG_Annotations.ECG_Valid.value): 
    data, times = _make_ecg(
        raw, 
        start = tmin, 
        stop = tmax, 
        reject_by_annotation = reject_by_annotation)
    
    onsets, ends = _annotations_start_stop_improved(
        raw, 
        annotations_to_keep = None, 
        annotations_to_reject = "bad", 
        tmin = tmin, 
        tmax = tmax)

    if annotate_valid_periods is not None: 
        write_to_annotations(raw, onsets, ends, annotate_valid_periods)
    if ecg_ch_name is not None: 
        indices = raw.time_as_index(times)
        ecg_data = np.zeros((1, raw.n_times), dtype = float)
        ecg_data[indices] = data.flatten()
        _add_data_to_raw(raw, ecg_data, ecg_ch_name, "ecg")
    return data, times


def _select_single_ecg_channel(
        raw, 
        ch_name: str | None = None, 
        return_data = False):
    if ch_name is None: 
        ch_name = ECG_Channels.ECG_Clean.value 
        if ch_name not in raw.ch_names: 
            ch_name = ECG_Channels.ECG_Raw.value
        if ch_name not in raw.ch_names: 
            raise ValueError("No ECG Channels found")
    _validate_ch_name_and_type(raw, ch_name, "ecg")
    idx_ecg = raw.ch_names.index(ch_name)
    if return_data: 
        ecg = raw.get_data(picks = idx_ecg)[0]
        return idx_ecg, ecg
    return idx_ecg


def _validate_ch_name_and_type(
    raw: mne.io.BaseRaw, 
    ch_name: str | None, 
    type: str = "ecg"      
) -> None:
    if ch_name is None: 
        raise ValueError("Please Enter a Channel Name")
    type_chans = mne.channel_indices_by_type(raw.info)[type]
    if ch_name not in raw.ch_names: 
        raise ValueError(f"The Given Channel {ch_name} was not found")
    ecg_chan_index = raw.ch_names.index(ch_name)
    if ecg_chan_index not in type_chans: 
        raise ValueError(f"Channel {ch_name} is not an {type} channel")


def _ch_name_and_type_in_raw(
    raw: mne.io.BaseRaw, 
    ch_name: str | None, 
    type: str = "ecg"  
): 
    return not (
        raw.ch_names is None or 
        ch_name not in raw.ch_names or 
        raw.ch_names.index(ch_name) not in mne.channel_indices_by_type(raw.info)[type]
    )
    


def load_ecg(
        raw: mne.io.BaseRaw,
        ecg: np.ndarray | None = None,
        ch_name: str | None = None, 
) -> np.ndarray: 
    if ecg is None:
        _, ecg = _select_single_ecg_channel(
            raw, 
            ch_name, 
            return_data = True
        )
    return ecg

def load_ecg_peaks(
        raw: mne.io.BaseRaw | None = None, 
        peak_ch_name: str | None = None, 
        ecg_ch_name: str | None = None,
        events: np.ndarray | None = None, 
        event_id: int | list[str] | None = None, 
        force_ch_name: bool = False): 
    #Load it from raw
    if events is None: 
        if peak_ch_name is None: 
            if force_ch_name:
                raise ValueError("Please Enter a Peak Channel Name")
            peak_ch_name = ECG_Channels.ECG_R_Peaks.value
        if force_ch_name or _ch_name_and_type_in_raw(raw, peak_ch_name, "ecg"): 
            ecg_peak_data = raw.get_data(picks = peak_ch_name, return_times = False)
        else: 
            print(f"Channel {peak_ch_name} not found, Exception: {e}")
            print("Attempting to extract channel data")
            ecg_data = load_ecg(
                raw, 
                peak_ch_name = ecg_ch_name)
            # Now run the Peak Extraction
            from mne.brainheart.ecg_wrappers import find_ecg_events_neurokit
            
            ecg_peak_data = find_ecg_events_neurokit(
                raw, 
                peak_ch_name = ecg_ch_name,
                keep_by_annotations = ECG_Annotations.ECG_Valid.value,
                reject_by_annotations = ["bad", "edge"]
            )
        # Turn this matrix into an Events array
        indices = np.where(ecg_peak_data)[1]
        events = np.hstack([
            indices[:, None], np.zeros((len(indices), 1), dtype = int), ecg_peak_data[:, indices].astype(int).T
        ])
    if event_id is not None: 
        if isinstance(event_id, int):
            events = events[events[:, 2] == event_id]
        elif isinstance(events, list):
            events = events[
                np.any(np.stack(
                    [events[:, 2] == id for id in events], axis = 0
                ), axis = 0)
            ]
    return events

def load_ecg_peaks_corrected(
        raw: mne.io.BaseRaw | None = None, 
        peak_corrected_ch_name: str = ECG_Channels.ECG_R_Peaks_Corrected.value,
        peak_ch_name: str | None = None, 
        ecg_ch_name: str | None = None,
        events: np.ndarray | None = None, 
        event_id: int | list[str] | None = None, 
): 
    if not _ch_name_and_type_in_raw(raw, peak_corrected_ch_name, "ecg"): 
        from mne.brainheart.ecg_wrappers import ecg_fixpeaks_neurokit
        ecg_fixpeaks_neurokit(
            raw, 
            ecg_peaks_ch_name = peak_ch_name, 
            ecg_peaks_corrected_ch_name = peak_corrected_ch_name
        )
    ecg_peak_data = raw.get_data(
        picks = raw.ch_names.index(peak_corrected_ch_name), return_times = False
    )
    indices = np.where(ecg_peak_data)[1]
    events = np.hstack([
        indices[:, None], np.zeros((len(indices), 1), dtype = int), ecg_peak_data[:, indices].astype(int).T
    ])
    return events


def load_hr(
        raw: mne.io.BaseRaw, 
        hr: np.ndarray | None = None,
        events: np.ndarray | None = None,
        event_id: int | None = None,
        ecg_ch_name: str | None = ECG_Channels.ECG_Clean.value,
        peak_ch_name: str | None = ECG_Channels.ECG_R_Peaks.value,
        hr_ch_name: str | None = ECG_Channels.ECG_Rate.value
): 
    if hr is None: 
        if _ch_name_and_type_in_raw(raw, hr_ch_name, "ecg"):
            hr = raw.get_data(picks = hr_ch_name, return_times = False)
        else:  
            hr = hr_neurokit2(
                raw, 
                events = events, 
                event_id = event_id, 
                ch_name = ecg_ch_name, 
                peak_ch_name = peak_ch_name)
    return hr


def load_hr_corrected(
        raw: mne.io.BaseRaw, 
        hr: np.ndarray | None = None,
        events: np.ndarray | None = None,
        event_id: int | None = None,
        ecg_ch_name: str | None = ECG_Channels.ECG_Clean.value,
        peak_ch_name: str | None = ECG_Channels.ECG_R_Peaks.value,
        hr_ch_name: str | None = ECG_Channels.ECG_Rate.value
): 
    if hr is None: 
        if _ch_name_and_type_in_raw(raw, hr_ch_name, "ecg"):
            hr = raw.get_data(picks = hr_ch_name, return_times = False)
        else:  
            hr = hr_neurokit2(
                raw, 
                events = events, 
                event_id = event_id, 
                ch_name = ecg_ch_name, 
                peak_ch_name = peak_ch_name)
    return hr

def load_ecg_quality(
        raw: mne.io.BaseRaw,
        ecg_quality: np.ndarray | None = None, 
        ecg_quality_ch_name: str | None = ECG_Channels.ECG_Quality.value
): 
    if ecg_quality is None: 
        _validate_ch_name_and_type(raw, ecg_quality_ch_name)
        ecg_quality = raw.get_data(picks = ecg_quality_ch_name, return_times = False)
    return ecg_quality
