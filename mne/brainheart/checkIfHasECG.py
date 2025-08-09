import mne
import os

datasets = [dataset for dataset in dir(mne.datasets) if not dataset.startswith("_")]

def list_files_recursive(path = "."):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(entry):
            list_files_recursive(full_path)
        else:
            print(full_path) 

#Alternative way
for dataset in os.listdir(r'C:/Users/azhao/mne_data'):
    if dataset.startswith("_"): 
        print(f"Skipping Dataset")
        continue
    print("dataset: ")
    list_files_recursive(os.path.join(r'C:/Users/azhao/mne_data', dataset))
    
#FOUND AN ECG CONCURRENT FILE

path = r"C:\Users\azhao\mne_data\MNE-epilepsy-ecog-data\sub-pt1\ses-presurgery\ieeg\sub-pt1_ses-presurgery_task-ictal_ieeg.eeg"
import mne_bids
import mne
import pandas as pd

bids_root = mne.datasets.epilepsy_ecog.data_path()

scans_file = os.path.join(bids_root, "\sub-pt1\ses-presurgery\sub-pt1_scans.tsv")
df = pd.read_csv(scans_file, sep='\t')
# Replace problematic date with a reasonable one
df['acq_time'] = df['acq_time'].str.replace('1920-07-24T19:35:19', '2020-07-24T19:35:19')
df.to_csv(scans_file, sep='\t', index=False)

subject = "pt1" #sample
bids_paths = mne_bids.BIDSPath(root = bids_root, 
                            subject = subject, 
                            extension = "vhdr"
                            )
bids_path = bids_paths.match()[0]
print(bids_path)
raw = mne_bids.read_raw_bids(bids_path)
print(raw.ch_names)

data1 = raw["EKG1", :][0]
data2 = raw["EKG2", :][0]
import matplotlib.pyplot as plt

plt.plot(data1[0]- data2[0])
plt.show()