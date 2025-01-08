import keras
from keras import ops
from keras.src.layers import Dense, Dropout, LSTM, Conv2D

def feed_forward (hidden_units,dropout_rate):
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(Dense(units, activation="relu"))
        fnn_layers.append(Dropout(dropout_rate))

    return keras.Sequential(fnn_layers)

def lstm_layer (hidden_lstm_units):
    lstm_layers = []
    for units in hidden_lstm_units[:-1]:
        lstm_layers.append(LSTM(units, return_sequences = True))
    lstm_layers.append(LSTM(hidden_lstm_units[-1]))

    return keras.Sequential(lstm_layers)

def conv_layer (filter_sizes):
    conv_layers = []
    for filter_size in filter_sizes:
        conv_layers.append(Conv2D(filters=filter_size, kernel_size=3, activation='relu', padding='same'))

    return keras.Sequential(conv_layers)


class BirdNet(keras.Model):
    def __init__(self,hidden_units, dropout_rate, lstm_hidden_units, bird_classes, filter_sizes, *args,**kwargs):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.bird_classes = bird_classes
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.lstm_hidden_units = lstm_hidden_units
        self.node_selector = Dense(1, activation="sigmoid")
        self.lstm_layers = lstm_layer(lstm_hidden_units)
        self.postprocessor = feed_forward(hidden_units, dropout_rate)
        self.compute_logits = Dense(bird_classes)
        self.conv = conv_layer(filter_sizes)

    def call(self, inputs):

        input_shape = ops.shape(inputs)
        batch_size, time_steps, features, channel = (input_shape[0], input_shape[1], input_shape[2], input_shape[3])

        x1 = self.conv(inputs)
        x2 = self.node_selector(x1)
        x3 = x2 * inputs
        x4 = ops.reshape(x3, (batch_size, time_steps, features * channel))

        return self.compute_logits(x4)

    def build(self, input_shape):
        super(BirdNet, self).build(input_shape)

    def get_config(self):
        return {"hidden_units": self.hidden_units,
                "lstm_hidden_units": self.lstm_hidden_units,
                "dropout_rate": self.dropout_rate,
                "bird_classes": self.bird_classes,
                "filter_sizes": self.filter_sizes}

