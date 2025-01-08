import os
import numpy as np
import waveletdecomp
import create_dataset
from modelmultilabel import BirdNet
from os.path import join as pjoin

dir_name = os.path.dirname(__file__)
data_dir = pjoin(dir_name, 'predict')
model_load = pjoin(dir_name, "weights", "zenodo_model2conv.weights.h5")
dropout_rate = 0.1
hidden_units = [128, 128]
lstm_hidden_units = [128,128]
packets_to_display = 10
bird_classes = 48
filter_sizes = [32, 32]

predict_data = []

for file_name in os.listdir(data_dir):
    if file_name.endswith((".WAV", ".wav")):
        print(f"Found {file_name}")

        file_path = os.path.join(data_dir, file_name)

        # Segment the file
        data = waveletdecomp.waveletsegment(file_path)

        #For each second
        for i in range(0, data.shape[0]):
            # Decompose the wav file
            packet = waveletdecomp.wavpacketdecomp(data[i])

            #Collect the coefficients
            _, _, nodes = waveletdecomp.collect_coefficients(packet)

            #Append to list
            predict_data.append(nodes)

predict_data = np.array(predict_data)
predict_data = predict_data.reshape(1, *predict_data.shape, 1)

#Initialize the model
model = BirdNet(
    hidden_units = hidden_units,
    lstm_hidden_units = lstm_hidden_units,
    dropout_rate = dropout_rate,
    bird_classes = bird_classes,
    filter_sizes= filter_sizes,
    name="target_bird_filter"
)

#Compile the model and load weights
model.build(predict_data.shape)
model.load_weights(model_load)

#Predict
#results = model.predict(predict_data)

#Predict for model 2
results = model.predict(predict_data)

#Print results
create_dataset.read_non_normalized_results(pjoin(dir_name, "combined_dataset", "combined_dataset_with_call_length.csv"), results, 0.9)
