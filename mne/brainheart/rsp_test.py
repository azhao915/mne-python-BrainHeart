import neurokit2 as nk
import mne

def simulate_dummy_raw_with_rsp(length_sec: int = 10, 
                                fs: int = 1000, 
                                noise: float = 0.01, 
                                respiratory_rate: int = 15, 
                                method = "breathmetrics") -> mne.io.BaseRaw: 
    
    simulated_rsp = nk.rsp_simulate(duration = length_sec, 
                                    sampling_rate = fs, 
                                    noise = noise, 
                                    respiratory_rate = respiratory_rate, 
                                    method = method) # (N, )
    simulated_rsp = simulated_rsp[None, :]    
    rsp_info = mne.create_info(ch_names = ["rsp"], sfreq = fs, ch_types = ["resp"])
    return mne.io.RawArray(
        data = simulated_rsp, 
        info = rsp_info
    )

if __name__ == "__main__": 
    from mne.brainheart.resp_wrappers import resp_process_neurokit

    rsp_raw_array = simulate_dummy_raw_with_rsp()
    print(resp_process_neurokit(rsp_raw_array))
