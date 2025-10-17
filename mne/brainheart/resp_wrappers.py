import mne
from mne.utils import logger
import neurokit2 as nk

import pandas as pd
import numpy as np

from utils import _write_events_dict_to_stim, _add_data_to_raw
from ecg_wrappers import _select_single_ecg_channel, _load_hr
from annotations_utils import _annotations_start_stop_improved, _onsets_ends_to_intervals


def resp_process_neurokit(
        raw: mne.io.BaseRaw, 
        resp_ch_name: str | None = None,
        **kwargs
): 
    _, rsp_data = _select_single_resp_channel(raw, resp_ch_name, return_data = True)
    resp_dict, resp_event_indices = nk.rsp_process(rsp_data, sampling_rate = raw.info["sfreq"], **kwargs)
    ch_types = neurokit_ch_names_to_types(resp_dict.columns)
    _write_events_dict_to_stim(resp_dict, raw, ch_types = ch_types)
    return raw, resp_event_indices, resp_dict


def _select_single_resp_channel(raw, ch_name: str = None, return_data = False): 
    idx_resp = _get_resp_channel_index(ch_name, raw)
    if idx_resp is not None:
        logger.info(f"Using channel {raw.ch_names[idx_resp]} to identify Respiratory Data.")
    else: 
        raise ValueError(
            "No Resp Channel Found"
        )
    ########### HERE, instead of raising a Value Error, EXTRACT IT FROM THE ECG
    if return_data: 
        resp = raw.get_data(picks = idx_resp)[0]
        return idx_resp, resp
    return idx_resp


def _get_resp_channel_index(ch_name, inst):
    """Get RESP channel index, if no channel found returns None."""
    #Copied Directly off of _get_ecg_channel_index
    if ch_name is None:
        resp_idx = mne.pick_types(
            inst.info,
            meg=False,
            eeg=False,
            stim=False,
            eog=False,
            ecg=False,
            resp = True,
            emg=False,
            ref_meg=False,
            exclude="bads",
        )
    else:
        if ch_name not in inst.ch_names:
            raise ValueError(f"{ch_name} not in channel list ({inst.ch_names})")
        resp_idx = mne.pick_channels(inst.ch_names, include=[ch_name])
    if not len(resp_idx):
        return None
    if len(resp_idx) > 1:
        print(
            f"More than one Resp channel found. Using only {inst.ch_names[resp_idx[0]]}."
        )
    return resp_idx[0]
        

def neurokit_ch_names_to_types(
        ch_names: str | list[str],
):
    
    
    def _neurokit_name_to_type(
            name: str
    ): 
        name = name.lower()
        if "raw" in name or "clean" in name: 
            return "resp"
        if "rate" in name: 
            return "resp" # Will need to modify this at some point
        return "stim"    
    
    
    if not len(ch_names): 
        return []
    if isinstance(ch_names, str): 
        ch_names = [ch_names]
    elif isinstance(ch_names, pd.Series): 
        ch_names = list(ch_names)
    return [_neurokit_name_to_type(name) for name in ch_names]


def resp_from_ecg_neurokit(
        raw: mne.io.BaseRaw | None = None, 
        hr: np.ndarray | None = None,
        events: np.ndarray | None = None,
        event_id: int | None = None,
        tstart: float | int = 0.0, 
        tend: float | int | None = None,
        min_segment_time: int | float | None = None,
        annotations_to_keep: str | list[str] | None = ["ecg_valid", "ecg_acceptable"],
        annotations_to_reject: str | list[str] | None = ["bad", "edge"], 
        method = "vangent2019",
        resp_name: str | None = None,
        verbose = True,
): 
    sfreq = raw.info["sfreq"]
    hr = _load_hr(raw, hr, events, event_id, "ECG_Rate")
    onsets, ends = _annotations_start_stop_improved(
        raw = raw,
        annotations_to_keep = annotations_to_keep, 
        annotations_to_reject = annotations_to_reject,
        tmin = tstart, 
        tmax = tend, 
        min_segment_time = min_segment_time,
        verbose = verbose
    )
    if not len(onsets):
        #Then have found no appropriate segments
        print("No segments were appropriate for ECG Peak Extraction")
        return None
    resp = np.zeros((1, raw.n_times), dtype = float)
    for (onset, end) in zip(onsets, ends): 
        hr_win = hr[onset:end]
        resp_win = nk.ecg_rsp(hr_win, sfreq, method = method)
        resp[0, onset:end] = resp_win
    if resp_name is not None: 
        _add_data_to_raw(raw, resp, resp_name, ch_types = "resp")
    return resp


if __name__ == "__main__": 

    from brainheart.rsp_test import simulate_dummy_raw_with_rsp
    rsp_raw_array = simulate_dummy_raw_with_rsp(100)
    print(resp_process_neurokit(rsp_raw_array))

