import mne
import numpy as np

from mne.preprocessing.ecg import _get_ecg_channel_index, _make_ecg

from mne.brainheart.annotations_utils import _annotations_start_stop_improved

from mne.brainheart.ecg_annotations_enum import ECG_Annotations
from mne.brainheart.utils import _add_data_to_raw

from ecg_channel_names_enum import ECG_Channels


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
        sfreq = raw.info["sfreq"]
        annotation_valid = mne.Annotations(
            onset = onsets/sfreq,
            duration = (ends - onsets)/sfreq,
            description = annotate_valid_periods, 
        )   
        raw.set_annotations(raw.annotations + annotation_valid)
    if ecg_ch_name is not None: 
        indices = raw.time_as_index(times)
        ecg_data = np.zeros((1, raw.n_times), dtype = float)
        ecg_data[indices] = data.flatten()
        _add_data_to_raw(raw, ecg_data, ecg_ch_name, "ecg")
    return data, times


def _select_single_ecg_channel(
        raw, 
        ch_name: str | None = ECG_Channels.ECG_Clean.value, 
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
    ch_name: str, 
    type: str = "ecg"      
) -> None:
    type_chans = mne.channel_indices_by_type(raw.info)[type]
    if ch_name not in raw.ch_names: 
        raise ValueError(f"The Given Channel {ch_name} was not found")
    ecg_chan_index = raw.ch_names.index(ch_name)
    if not ecg_chan_index == type_chans[0]: 
        raise ValueError(f"Channel {ch_name} is not an {type} channel")


def _load_hr(
        raw: mne.io.BaseRaw, 
        hr: np.ndarray | None = None,
        events: np.ndarray | None = None,
        event_id: int | None = None,
        ecg_ch_name: str | None = ECG_Channels.ECG_Clean.value,
        hr_ch_name: str | None = ECG_Channels.ECG_Rate.value
): 
    if hr is None: 
        if hr_ch_name in raw.ch_names and hr_ch_name is not None:
            hr = raw.get_data(picks = hr_ch_name, return_times = False)
        else:  
            hr = hr_neurokit2(raw, events = events, event_id = event_id, ch_name = ecg_ch_name)
    return hr


def load_ecg_quality(
        raw: mne.io.BaseRaw,
        ecg_quality: np.ndarray | None = None, 
        ecg_quality_ch_name: str | None = ECG_Channels.ECG_Quality.value
): 
    if ecg_quality is None: 
        if ecg_quality_ch_name is None: 
            raise ValueError("Please Enter a Value for Either the ECG Quality or the ECG Quality Channel Name")
        ecg_quality = raw.get_data(picks = ecg_quality_ch_name, return_times = False)
    return ecg_quality
