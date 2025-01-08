import math
import os
import numpy as np
import pandas as pd
import waveletdecomp
from sklearn.preprocessing import MultiLabelBinarizer
from os.path import join as pjoin


def convert_annotatfiles_to_list(annot_file_path, segment_length, dataset_filepath):
    recording_data = []

    #Import dataset
    dataset_df = read_csv(dataset_filepath)

    for annot_record_no in os.listdir(annot_file_path):
        for recording_segment in os.listdir(pjoin(annot_file_path, annot_record_no)):
            annotation_segment = np.zeros(shape=segment_length, dtype="object")

            if recording_segment.endswith(".selections.txt"):
                print(f"Opened {recording_segment} in {annot_record_no}")

                rec_name_and_seg = recording_segment.split('.')[0]

                #Isolate the appropriate rows
                if dataset_df[dataset_df["Record_Segment"] == rec_name_and_seg].empty:
                    recording_data.append(annotation_segment)
                    continue
                rows = dataset_df[dataset_df["Record_Segment"] == rec_name_and_seg]

                #Write for each segment
                for row in rows.iterrows():
                    for seconds in range(math.ceil(row[1]["Call_time"])):

                        if annotation_segment[math.floor(row[1]["Begin Time (s)"]) + seconds] == 0:
                            annotation_segment[math.floor(row[1]["Begin Time (s)"] + seconds)] = row[1]["Species"]
                        elif row[1]["Species"] in annotation_segment[math.floor(row[1]["Begin Time (s)"]) + seconds]:
                            continue
                        else:
                            annotation_segment[math.floor(row[1]["Begin Time (s)"]) + seconds] += " " + row[1]["Species"]

            recording_data.append(annotation_segment)

    return recording_data

def read_csv(csv_filepath):
    return pd.read_csv(csv_filepath)

def create_dataset_for_segment(parent_sound_file_path):
    """args: accepts the folder path which has the recording in order.
    Make sure to only use this function when the wav files are of the exact same size, otherwise, during the conversion to numpy arrays,
    an inhomogeneous error will be thrown."""

    wav_decomposed = []

    # List all files in the parent recording folder
    for recording_no in os.listdir(parent_sound_file_path):
        for recording_file in os.listdir(pjoin(parent_sound_file_path, recording_no)):
            if recording_file.endswith((".WAV", ".wav")):
                print(f"Found {recording_file} in {recording_no}")

                # Segment the file
                data = waveletdecomp.waveletsegment(pjoin(parent_sound_file_path, recording_no, recording_file))

                temp_node_data = []

                # For each second
                for i in range(0, data.shape[0]):

                    # Decompose the wav file
                    packet = waveletdecomp.wavpacketdecomp(data[i])

                    # Collect the coefficients
                    _, _, nodes = waveletdecomp.collect_coefficients(packet)

                    # Append numpy
                    temp_node_data.append(nodes)

                wav_decomposed.append(temp_node_data)

    wav_decomposed = np.array(wav_decomposed)

    return wav_decomposed.reshape(*wav_decomposed.shape, 1)

def check_sound_folder(sound_file_path):
    sound_file_list = []
    sound_length = []

    try:
        for recording_no in os.listdir(sound_file_path):
            for recording_file in os.listdir(pjoin(sound_file_path, recording_no)):
                if recording_file.endswith((".WAV", ".wav")):
                    sound_file_list.append(recording_file)
                    data = waveletdecomp.waveletsegment(pjoin(sound_file_path, recording_no, recording_file))
                    sound_length.append(data.shape[0])
                else:
                    raise TypeError("Cannot have files in any other format than .wav / .WAV")
    except Exception as e:
        print(f"Found exception: {e}")

    try:
        # Ensure the list is not empty
        assert sound_length, "The list is empty."
        first_value = sound_length[0]
        assert all(value == first_value for value in sound_length), \
            f"Not all values are the same, program will crash. Found values: {sound_length}"
        print("All values in the list are the same.")
    except Exception as e:
        print(f"Assertion failed: {e}")

def convert_species_column_to_onehot(df_dataset ,np_dataset):

    #Collect unique species
    fit_transform_array = df_dataset["Species"].unique()

    #Instantiate the one - hot encoder class and fit  # Define the category order
    encoder = MultiLabelBinarizer()
    encoder.fit(fit_transform_array.reshape(-1, 1))

    #Iterate through the numpy dataset
    annotated_numpy_encoded_shape = (*np_dataset.shape, fit_transform_array.shape[0])
    annotated_numpy_encoded = np.zeros(shape=annotated_numpy_encoded_shape, dtype="int")
    for segindex, segments in enumerate(np_dataset):
        for spindex, species in enumerate(segments):
            if species == 0:
                continue

            #Collect species list and transform
            species_lis = np.array(species.split(" "))
            species_lis = species_lis.reshape(1, -1)
            annotated_numpy_encoded[segindex, spindex] = encoder.transform(species_lis)

    return annotated_numpy_encoded

def read_non_normalized_results(dataset_filepath, results, threshold):
    #Import dataset
    dataset_df = read_csv(dataset_filepath)
    index_array = dataset_df["Species"].unique()
    results = 1 / (1 + np.exp(-results))

    for indext, time_segment in enumerate(results[0]):
        for index, bird_specie_prob in enumerate(time_segment):
            if bird_specie_prob > threshold:
                print(f"for {indext} found {index_array[index]}")

if __name__ == "__main__":
    #annotation_file = pjoin(os.path.dirname(__file__), "annotation_files")
    #annotated_dataset = pjoin(os.path.dirname(__file__), "combined_dataset", "combined_dataset_with_call_length.csv")
    #annotated_save_encoded = pjoin(os.path.dirname(__file__), "combined_numpy_dataset", "annotated_numpy_data_unencoded.npy")
    #combined_dataset_numpy = np.array(convert_annotatfiles_to_list(annot_file_path=annotation_file,
                                                                   #segment_length=300, dataset_filepath=annotated_dataset))

    #Save the annotation data [rows for each segment]
    #waveletdecomp.savenumpy(data_save, combined_dataset_numpy)

    #Prelim sound folder check
    #wav_file_location = pjoin(os.path.dirname(__file__), "Recordings")
    #check_sound_folder(wav_file_location)

    #Save the wav sound dataset [rows for each segment]
    sound_folder_path = pjoin(os.path.dirname(__file__), "Recordings")
    sound_folder_save = pjoin(os.path.dirname(__file__), "combined_numpy_dataset", "sound_dataset_encoded.npy")
    sound_dataset = create_dataset_for_segment(sound_folder_path)
    waveletdecomp.savenumpy(sound_folder_save, sound_dataset)

    #Encode the annotation file
    #annotated_encoded = pjoin(os.path.dirname(__file__), "combined_numpy_dataset", "annotated_numpy_data_encoded.npy")
    #df_annotated_dataset = read_csv(annotated_dataset)
    #np_annotated_array = np.load(annotated_save_encoded, allow_pickle=True)
    #annotated_dataset_encoded = convert_species_column_to_onehot(df_annotated_dataset, np_annotated_array)

    #Save the numpy
    #waveletdecomp.savenumpy(annotated_encoded, annotated_dataset_encoded)


