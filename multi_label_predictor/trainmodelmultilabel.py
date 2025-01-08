import os
import keras
import tensorflow
import numpy as np
import callbackmetric
from matplotlib import pyplot as plt
from os.path import join as pjoin
from modelmultilabel import BirdNet
from recall_specificity_weights_loss import custom_loss


dir_name = os.path.dirname(__file__)
model_save = pjoin(dir_name, "weights")
data_targets = pjoin(dir_name, 'combined_numpy_dataset', "annotated_numpy_data_encoded.npy")
data_load= pjoin(dir_name, 'combined_numpy_dataset', "sound_dataset_encoded.npy")
bird_features = np.load(data_load)
bird_targets = np.load(data_targets)
dropout_rate = 0.1
learning_rate = 0.01
hidden_units = [256, 256]
lstm_hidden_units = [256,256]
bird_classes = 48
filter_sizes = [32, 32]
epochs = 20
batch_size = 10


early_stopping = keras.callbacks.EarlyStopping(
        monitor="acc", patience=50, restore_best_weights=True
)

#Initialize the model
model = BirdNet(
    hidden_units = hidden_units,
    lstm_hidden_units = lstm_hidden_units,
    dropout_rate = dropout_rate,
    bird_classes = bird_classes,
    filter_sizes= filter_sizes,
    name="target_bird_filter"
)

#Compile the model
model.compile(
        optimizer=keras.optimizers.Adam(learning_rate= learning_rate),
        loss=custom_loss(0.8,0.2, from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc"), "true_positives", "true_negatives", "false_positives", "false_negatives"]
)

#Train the model
metrics_logger = callbackmetric.MetricsLogger()
model.fit(
        shuffle = False,
        x=bird_features,
        y=bird_targets,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, metrics_logger],
        validation_split=0.4
)

#Save model weights
#model.save(f"{model_save}/zenodo_model2conv.weights.h5")

#Plot TP, FP, FN, TN
epochs = range(1, len(metrics_logger.tp) + 1)

plt.figure(figsize=(10, 6))

plt.plot(epochs, metrics_logger.tp, label='True Positives')
plt.plot(epochs, metrics_logger.fp, label='False Positives')
plt.plot(epochs, metrics_logger.tn, label='True Negatives')
plt.plot(epochs, metrics_logger.fn, label='False Negatives')

plt.title('Training Metrics Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()
