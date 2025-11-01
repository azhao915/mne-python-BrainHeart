import mne
import numpy as np

#from mne.brainheart.loading import _load_hr
from mne.time_frequency.psd import psd_array_welch
from mne.brainheart.annotations_utils import _annotations_start_stop_improved

import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.stats import norm


def jackknife_CI_percentile(jackknife_psds, psd, alpha = 0.95): 
    lower_q = (1-alpha)/2
    upper_q = 1-lower_q
    lower = np.percentile(jackknife_psds, q = lower_q*100, axis = -1)
    upper = np.percentile(jackknife_psds, q = upper_q*100, axis = -1)
    return lower, upper

def jackknife_CI_z_score(jackknife_psds, psd, alpha = 0.95): 
    n = jackknife_psds.shape[-1]
    jackknife_var = (n-1)/n*np.sum(np.square(jackknife_psds - psd[..., np.newaxis]), axis = -1)
    jackknife_se = np.sqrt(jackknife_var)
    CI_width = norm.ppf((1-alpha)/2 + alpha)*jackknife_se
    return psd - CI_width, psd + CI_width


def welch_with_CI_from_raw(
        raw: mne.io.BaseRaw, 
        picks: list[int] | None = None, 
        psd_func = psd_array_welch,
        fmax = 100,
        dB = False,
        CI_func = jackknife_CI_z_score,
        alpha = 0.95,
        n_jobs = -1,
        annotations_to_reject: str | list[str] | None = None, 
        annotations_to_keep: str | list[str] | None = None, 
        tmin = 0, 
        tmax = None, 
        min_segment_time = 10.00, 
        verbose = True
):
    fs = raw.info["sfreq"] 
    onsets, ends = _annotations_start_stop_improved(
        raw, 
        annotations_to_keep = annotations_to_keep, 
        annotations_to_reject = annotations_to_reject, 
        tmin = tmin, 
        tmax = tmax, 
        min_segment_time = min_segment_time, 
        verbose = verbose
    )

    freqs, psds = get_psds(
        raw = raw,
        picks = picks,
        psd_func = psd_func,
        fs = fs, 
        onsets = onsets,
        ends = ends, 
        n_jobs = n_jobs, 
        dB = dB, 
        fmax = fmax)
    
    return welch_with_CI(freqs, psds, CI_func, alpha)
    
def welch_with_CI(freqs, psds, CI_func = jackknife_CI_z_score, alpha = 0.95):
    jackknife_psds = get_jackknife_psds(psds)
    psd = np.mean(psds, axis = -1)
    jackknife_psd = np.mean(jackknife_psds, axis = -1)
    bias = psd - jackknife_psd

    lower, upper = CI_func(jackknife_psds, psd, alpha)

    return freqs, psd, bias, lower, upper
    

def get_psds(raw, picks, psd_func, fs, onsets, ends, n_jobs, dB, fmax): 

    psds_to_concat = [] # Can do this easier, but for now just do this
    for onset, end in zip(onsets, ends): 
        data = raw.get_data(picks = picks, start = onset, stop = end)
        psds_period, freqs = psd_func(
            data, 
            sfreq = fs,
            n_fft = int(fs*3),
            fmax = fmax,
            average = False, 
            output = "power", 
            n_jobs = n_jobs)
        psds_to_concat.append(psds_period)
    
    psds = np.concat(psds_to_concat, axis = -1)
    if dB: 
        psds = 20*np.log10(psds)
    return freqs, psds

def get_jackknife_psds(psds):
    n = psds.shape[-1]
    summed_psds = np.sum(psds, axis = -1)
    jackknife_data = summed_psds[..., np.newaxis] - psds
    return jackknife_data/(n-1)

def plot_psd_with_CI(
        freqs,
        psd,
        channel_num,
        lower, 
        upper, 
        ax, 
        alpha: float = 0.5,
        unit: str = "V",
        dB: float = False, 
        ch_name: str = "Channel"):
    channel_num = channel_num if (isinstance(channel_num, list) or isinstance(channel_num, np.ndarray)) else [channel_num]
    for n in channel_num:
        ax.plot(freqs, psd[n, :].T)
        ax.fill_between(freqs, lower[n, :].T, upper[n, :].T, alpha = alpha)
    ax.set_xlabel("frequency (Hz)")
    if dB: 
        unit = unit + " dB"
    ax.set_ylabel(unit)
    ax.set_title(ch_name)
    ax.grid()



if __name__ == "__main__": 
    from load_reference_dataset import load
    raw = load()
    raw.load_data()

    dB = True

    freqs, psd, bias, lower, upper = welch_with_CI_from_raw(raw, fmax = 100, dB = dB)
    channel_num = np.arange(20)*5

    chan_name = "Channels"

    #plt.style.use("dark_background")

    fig, ax = plt.subplots()

    plot_psd_with_CI(
        freqs,
        psd,
        channel_num,
        lower, 
        upper, 
        ax, 
        ch_name = chan_name, 
        dB = dB)

    plt.show()