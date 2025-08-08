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
        event_id: int = 999, 
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
    peaks = np.concatenate(peaks)
    n_peaks = len(peaks)
    #Now Annotate the valid ecg periods
    if annotate_valid_ecg_period is not None:
        ecg_annotations = mne.Annotations(
            onset = onsets,
            duration = ends - onsets,
            description = annotate_valid_ecg_period
        )
        #Now add to existing annotations
        raw.set_annotations(raw.annotations + ecg_annotations)
    
    return np.stack([
            peaks, 
            np.zeros(n_peaks, dtype = int), 
            np.ones(n_peaks, dtype = int)*event_id
        ], axis = 1)


def ecg_clean_neurokit(
        raw, 
        ch_name = None, 
        method: str = "neurokit", 
        **kwargs
): 
    idx_ecg = _get_ecg_channel_index(ch_name, raw)
    if idx_ecg is not None:
        logger.info(f"Cleaning channel {raw.ch_names[idx_ecg]} with method {method}")
        #Added sfreq here, as needed for neurokit2.ecg_peaks
        sfreq = raw.info["sfreq"]
    else: 
        #The Neurokit2 functions are only tested against real ECG, not simulated
        #As such, we aren't going to apply this function to simulated data
        raise ValueError(
            "No ECG Channel Found"
        )

    raw.apply_function(
        _ecg_clean_with_params(sampling_rate=sfreq, method = method, **kwargs),
        picks = idx_ecg, 
        channel_wise = True
    )


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

def ecg_fixpeaks_neurokit(
        raw, 
        events: np.ndarray,
        ch_name: str | None = None,
        iterative: bool = False, #Might replace with **kwargs
): 
    ecg_indx = _select_single_ecg_channel(raw, ch_name)
    ecg = raw[ecg_indx, :][0]
    sfreq = raw.info["sfreq"]
    artifacts, peaks_clean = nk.signal_fixpeaks(
        peaks = events[:, 0], 
        sampling_rate = sfreq, 
        method = "kubios", 
        iterative = iterative
    )

    print(artifacts.keys())
    #print(artifacts["drrs"])



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
    ecg_clean_neurokit(raw)
    events = find_ecg_events_neurokit(raw)
    print(events)

    from misc import bipolar_automatic_ref_seeg

    #epochs = mne.Epochs(raw_bipolar.load_data(), events)
    #epochs.load_data()
    #epochs.average(method = "mean").pick(picks = ["seeg"]).plot()
    ecg_fixpeaks_neurokit(raw, events)

    import matplotlib.pyplot as plt

    plt.show()