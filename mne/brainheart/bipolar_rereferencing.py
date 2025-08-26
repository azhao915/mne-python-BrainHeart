import numpy as np
import mne

def bipolar_automatic_ref_seeg(
        raw: mne.io.BaseRaw, 
        exclude: str = "bads"
): 
    channelNames = raw.copy().pick(picks = "seeg", exclude = exclude).ch_names
    electrodeNames = list(set([channelName[:3] for channelName in channelNames]))
    anodes = []
    cathodes = []
    for electrode in electrodeNames: 
        channelNamesInElectrode = [channelName for channelName in channelNames if electrode in channelName]
        channelNamesInElectrode.sort() #Sort as Strings
        prevChanNum = np.inf
        prevChanName = ""
        for chanName in channelNamesInElectrode: 
            chanNum = int(chanName[3:])
            if chanNum == prevChanNum + 1: 
                anodes.append(prevChanName)
                cathodes.append(chanName)
            prevChanName = chanName
            prevChanNum = chanNum
    raw_bipolar = mne.set_bipolar_reference(raw.load_data(), anode = anodes, cathode = cathodes, drop_refs = True, copy = True)
    raw_bipolar.pick(picks = ["seeg"], exclude = raw.ch_names)
    return raw_bipolar