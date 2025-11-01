from mne.io import BaseRaw, RawArray
from mne import create_info
import numpy as np
from pandas import Series

import itertools

from mne.brainheart.annotations_utils import _onsets_ends_to_intervals

def _inter_peaks_from_windows(
        peaks: list[list[int]] | list[int],
        sfreq: int
) -> list[np.ndarray]:
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


def _peaks_from_intervals(intervals, events, event_id: int | None = None, flattened: bool = False) -> list[int] | list[list[int]]:
    if not len(intervals): 
        return [[]]
    if isinstance(events, list): 
        all_peaks = np.array(events)
    else: 
        if event_id is not None:  
            events = events[events[:, 2] == event_id]
        all_peaks = events[:, 0]
    peaks = [[]]*len(intervals)
    for i, (onset, end) in enumerate(intervals): 
        peaks_mask = (onset <= all_peaks) & (all_peaks <= end)
        peaks[i] = all_peaks[peaks_mask]
    if flattened: 
        peaks = list(itertools.chain.from_iterable(peaks))
    return peaks


def write_events_to_channel(
        events: np.ndarray, 
        raw: BaseRaw, 
        ch_name: str | None = None,
        ch_type: str = "ecg"): 
    if ch_name is None or events is None: 
        return
    data = np.zeros((1, raw.n_times), dtype = int)
    if events.ndim == 1: 
        if len(events):
            # Then just a bunch of indices
            data[0, events] = 1
    else: 
        for event_id in np.unique(events[:, 2]): 
            event_mask = events[:, 2] == event_id
            events_index = events[event_mask]
            data[0, events_index] = event_id
    return _add_data_to_raw(
        raw = raw, 
        data = data, 
        ch_names = ch_name, 
        ch_types = ch_type)


def _write_events_dict_to_stim(events_dict: dict, raw: BaseRaw, ch_types: list[str] | str | None = None): 
    data = events_dict.values
    if data.ndim == 1: 
        data = np.reshape(data, (len(data), 1))
    data = data.T
    ch_names = events_dict.keys()
    ch_names = list(ch_names)
    return _add_data_to_raw(raw, data, ch_names, ch_types)

def _add_data_to_raw(
        raw: BaseRaw, 
        data: np.ndarray, 
        ch_names: list[str], 
        ch_types: list[str] | str | None = None):
    
    if isinstance(data, Series): 
        data = data.values
    if data.ndim == 1: 
        data = data[None, :]
    if isinstance(ch_names, str): 
        ch_names = [ch_names]
    if ch_types is None: 
        ch_types = ["ecg"]*len(ch_names)
    elif isinstance(ch_types, str): 
        ch_types = [ch_types]*len(ch_names)
    new_info = create_info(ch_names, raw.info["sfreq"], ch_types = ch_types)
    new_raw = RawArray(data, new_info)
    return raw.add_channels([new_raw], force_update_info = True)


def write_to_channel(
        data: np.ndarray, 
        raw: BaseRaw, 
        ch_name: str | None = None, 
        ch_type: str = "ecg"): 
    if ch_name is None: 
        return raw
    new_info = create_info(
        [ch_name], 
        raw.info["sfreq"], 
        ch_types = [ch_type])
    if isinstance(data, Series): 
        data = data.values
        data = np.reshape(data, (1, len(data)))
    new_raw = RawArray(data, new_info)
    return raw.add_channels([new_raw], force_update_info = True)


def _mask_from_intervals(intervals, N): 
    mask = np.zeros(N, dtype = bool)
    for onset, end in intervals: 
        mask[onset:end] = True
    return mask


def _intervals_from_mask(mask): 
    if not np.any(mask): 
        return np.array([], int), np.array([], int)
    interval_starts = np.where(mask & np.concatenate([[True], ~mask[:-1]]))[0]
    interval_ends = np.where(mask & np.concatenate([~mask[1:], [True]]))[0]
    return _onsets_ends_to_intervals(interval_starts, interval_ends)
    