from enum import Enum

class ECG_Channels(Enum): 
    """ An Enumerations for the Types of Channels for ECG and their corresponding Channel Name Strings
    """
    ECG_Raw = "ECG_Raw"
    ECG_Clean = "ECG_Clean"
    ECG_Rate = "ECG_Rate"
    ECG_Quality = "ECG_Quality"
    ECG_R_Peaks = "ECG_R_Peaks"
    # Added after, for fixing the peaks
    ECG_R_Peaks_Corrected = "ECG_R_Peaks_Corrected" # For Computing NN Measures
    # Now add all the fixpeaks elements
    ECG_fixpeaks_ectopic = 'ECG_fixpeaks_ectopic'
    ECG_fixpeaks_missed = 'ECG_fixpeaks_missed'
    ECG_fixpeaks_extra = 'ECG_fixpeaks_extra'
    ECG_fixpeaks_longshort = 'ECG_fixpeaks_longshort'
    # The corrected Heart-Rate, or NN Rate
    ECG_Rate_Corrected = "ECG_Rate_Corrected"
    
    ECG_P_Peaks = "ECG_P_Peaks"
    ECG_P_Onsets = "ECG_P_Onsets"
    ECG_P_Offsets = "ECG_P_Offsets"
    ECG_Q_Peaks = "ECG_Q_Peaks"
    ECG_R_Onsets = "ECG_R_Onsets"
    ECG_R_Offsets = "ECG_R_Offsets"
    ECG_S_Peaks = "ECG_S_Peaks"
    ECG_T_Peaks = "ECG_T_Peaks"
    ECG_T_Onsets = "ECG_T_Onsets"
    ECG_T_Offsets = "ECG_T_Offsets"
    ECG_Phase_Atrial = "ECG_Phase_Atrial"
    ECG_Phase_Completion_Atrial = "ECG_Phase_Completion_Atrial"
    ECG_Phase_Ventricular = "ECG_Phase_Ventricular"
    ECG_Phase_Completion_Ventricular = "ECG_Phase_Completion_Ventricular"