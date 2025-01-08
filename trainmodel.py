import os
import keras
import numpy as np
from os.path import join as pjoin
from model import BirdNet
import tensorflow

species_name = "Black-capped Bulbul"

dir_name = os.path.dirname(__file__)
model_save = pjoin(dir_name, "weights")
data_targets = pjoin(dir_name, 'birddataset', species_name, species_name+'_targets.npy')
data_load= pjoin(dir_name, 'birddataset', species_name, species_name+'_dataset.npy')
bird_features = np.load(data_load)
bird_targets = np.load(data_targets)
dropout_rate = 0.1
learning_rate = 0.01
hidden_units = [256, 256]
lstm_hidden_units = [256,256]
epochs = 30
batch_size = 50

early_stopping = keras.callbacks.EarlyStopping(
        monitor="acc", patience=50, restore_best_weights=True
)

#Initialize the model
model = BirdNet(
    hidden_units = hidden_units,
    lstm_hidden_units = lstm_hidden_units,
    dropout_rate = dropout_rate,
    name="target_bird_filter"
)

#Compile the model
model.compile(
        optimizer=keras.optimizers.SGD(learning_rate= learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")]
)

#Train the model
model.fit(
        shuffle = False,
        x=bird_features,
        y=bird_targets,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
)

#Save model weights
#model.save(f"{model_save}/{species_name}model.weights.h5")


