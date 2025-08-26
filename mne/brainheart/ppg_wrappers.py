import neurokit2 as nk

import numpy as np
import pandas as pd

from mne.preprocessing.ecg import _get_ecg_channel_index
import mne
from mne.utils import logger, verbose
from mne.annotations import _annotations_starts_stops

from functools import partial

from annotations_utils import _annotations_start_stop_improved, _onsets_ends_to_intervals, _intervals_to_onsets_ends
from utils import _intervals_from_mask, _write_events_dict_to_stim, _peaks_from_intervals, _format_peaks
from event_detection import find_events, sliding_window_accept_reject
from ecg_wrappers import _select_single_ecg_channel, neurokit_ch_names_to_types


def ppg_process_neurokit(
        raw: mne.io.BaseRaw, 
        ppg_ch_name: str | None = None,
        **kwargs
): 
    _, ppg_data = _select_single_ppg_channel(raw, ppg_ch_name, return_data = True)
    ppg_dict, ppg_event_indices = nk.ppg_process(ppg_data, sampling_rate = raw.info["sfreq"], **kwargs)
    ch_types = neurokit_ch_names_to_types(ppg_dict.columns)
    _write_events_dict_to_stim(ppg_dict, raw, ch_types = ch_types)
    return raw, ppg_event_indices


def _select_single_ppg_channel(raw, ppg_ch_name, return_data): 
    # NEED TO IMPLEMENT THIS
    return _select_single_ecg_channel(raw, ch_name = ppg_ch_name, return_data = return_data)


@verbose
def find_ppg_events_neurokit(
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
    idx_ecg = _select_single_ppg_channel(raw, ch_name, return_data=False)
    nk_ecg_peaks_wrapper = partial(
        lambda ecg_segment, sfreq, method: nk.ppg_peaks(ecg_segment.flatten(), sampling_rate=sfreq, method=method)[1]["ECG_R_Peaks"],
        method=method
    )
    clean_func =  partial(
        lambda ecg_segment, sfreq, method: nk.ppg_clean(ecg_segment.flatten(), sampling_rate=sfreq, method=method),
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


def ppg_clean_neurokit(
        raw, 
        ch_name = None, 
        method: str = "biosppy", 
        **kwargs
): 
    sfreq = raw.info["sfreq"]
    idx_ppg = _select_single_ecg_channel(
        raw, 
        ch_name, 
        return_data=False
    )
    def _ppg_clean_with_params(
            sampling_rate: int|float, 
            method: str, 
            **kwargs
    ): 
        def inner_func(ppgSignal): 
            return nk.ppg_clean(ppgSignal.flatten(), sampling_rate = sampling_rate, method = method, **kwargs)
        return inner_func

    raw.apply_function(
        _ppg_clean_with_params(sampling_rate=sfreq, method = method, **kwargs),
        picks = idx_ppg, 
        channel_wise = True
    )


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

    ppg_process_neurokit(raw)
    print(raw.ch_names[-20:])