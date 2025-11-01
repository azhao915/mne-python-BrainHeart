from enum import Enum

class ECG_Annotations(Enum): 
    ''' Keeps track of the Names of the Different Annotations
    '''
    ECG_Valid = "ECG_Valid"
    bad_ECG = "bad_ECG"
    bad_ECG_ZHAO2018 = "bad_ECG_ZHAO2018"