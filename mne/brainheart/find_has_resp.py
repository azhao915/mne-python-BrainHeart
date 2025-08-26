import os
import pandas as pd
import mne_bids
import pathlib

def find_subjects_with_electrodes_bids(
        root: str, 
        column: str,  
        column_query: str, 
        *args) -> dict[str: list[str]]:
    """Locates The Subjects, and their corresponding sessions and runs containing the column_query in the column of their channels.tsv file

    Raises:
        Warning: _description_
        Warning: _description_
        Warning: _description_

    Returns:
        dict[str: list[str]]: A Dictionary with the key being the subjects, and the values being a list of sessions/runs with the column_query in the column
    """
    #Might need to change the suffix later to an argument
    suffix = "channels"
    file_query_suffix = f"_{suffix}.tsv"
    participants_file = os.path.join(root, "participants.tsv")
    if os.path.exists(participants_file): 
        df = pd.read_csv(participants_file, sep = "\t")
        subjects = df.participant_id
    else: 
        print(f"No participants.tsv file found, directly scanning for subject directories")
        subjects = [directory for directory in os.listdir(root) if os.path.isdir(directory) and directory.startswith("sub-")]
    subjects_with_resp = dict()
    for subject in subjects: 
        subject_path = os.path.join(root, subject, *args)
        subject_channels_files = []
        if os.path.exists(subject_path):
            subject_channels_files = list(pathlib.Path(subject_path).rglob("*" + file_query_suffix))
            if not len(subject_channels_files): 
                raise Warning(f"No Channel File found for Subject {subject}")
        else: 
            raise Warning(f"Subject {subject} has no directory {subject_path}")
        for file in subject_channels_files: 
            subject_df = pd.read_csv(
                os.path.join(subject_path, file), sep = "\t"
            )
            if column not in subject_df.columns: 
                raise Warning(f"File {os.path.basename(file)} has not column {column}")
            #if column_query in list(subject_df[column]):
            if subject_df[column].str.startswith(column_query).any():
                if subject not in subjects_with_resp.keys(): 
                    subjects_with_resp[subject] = []
                subjects_with_resp[subject].append(
                    os.path.basename(file).removesuffix(file_query_suffix).removeprefix(subject + "_")
                )
    return subjects_with_resp

if __name__ == "__main__": 
    bids_root = r"D:\DABI\StimulationDataset"
    print("The Following Subjects Have Respiratory Data")
    for k, v in find_subjects_with_electrodes_bids(bids_root, column = "name", column_query = "RESP").items(): 
        print(f"{k}: {v}")
    print("The Following Subjects Have ECG Data")
    for k, v in find_subjects_with_electrodes_bids(root = bids_root, column_query = "ECG", column = "type").items(): 
        print(f"{k}: {v}")