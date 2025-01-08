import os
import pandas as pd
from os.path import join as pjoin

def convert_annotatfiles_to_list(parent_file_path):
    recording_data = []

    for file_name in os.listdir(parent_file_path):
        for recording_annotation in os.listdir(pjoin(parent_file_path,file_name)):
            if recording_annotation.endswith(".selections.txt"):
                print(f"Opened {recording_annotation}")

                with open(pjoin(parent_file_path,file_name,recording_annotation)) as annot_file:
                    for line in annot_file:
                        if line.startswith("Selection"):
                            continue
                        rec_name_and_seg = recording_annotation.split('.')[0]
                        rows = line.strip().split("\t")
                        rows.append(rec_name_and_seg)
                        recording_data.append(rows)

    return recording_data

def convert_annot_list_to_dataframe(recording_dataset):
    columns = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)",
               "Species","Record_Segment"]
    return pd.DataFrame(recording_dataset, columns=columns)

def save_as_csv(data_save, data_frame):
    data_frame.to_csv(pjoin(data_save,"combined_dataset.csv"), sep=",", index=False, encoding="utf-8")

if __name__ == "__main__":
    dir_name = os.path.dirname(__file__)
    annot_file_path = pjoin(dir_name, "annotation_files")
    annot_data_save = pjoin(dir_name, "combined_dataset")
    df = convert_annot_list_to_dataframe(convert_annotatfiles_to_list(parent_file_path=annot_file_path))

    #Save the dataframe
    #save_as_csv(data_save=annot_data_save,data_frame=df)
