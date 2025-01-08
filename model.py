import keras
from keras.src.layers import Dense, Dropout, LSTM, Flatten


def feed_forward (hidden_units,dropout_rate):
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.append(Dense(units, activation="relu"))
        fnn_layers.append(Dropout(dropout_rate))

    return keras.Sequential(fnn_layers)

def lstm_layer (hidden_lstm_units):
    lstm_layers = []
    for units in hidden_lstm_units:
        lstm_layers.append(LSTM(units, return_sequences = True))

    return keras.Sequential(lstm_layers)

class BirdNet(keras.Model):
    def __init__(self,hidden_units,dropout_rate,lstm_hidden_units,*args,**kwargs):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.lstm_hidden_units = lstm_hidden_units
        self.node_selector = Dense(1, activation="sigmoid")
        self.lstm_layers = lstm_layer(lstm_hidden_units)
        self.flatten = Flatten()
        self.postprocessor = feed_forward(hidden_units, dropout_rate)

        self.compute_logits = Dense(1)

    def call(self, inputs):
        x1 = self.lstm_layers(inputs)
        x2 = self.node_selector(x1)
        x3 = x2 * inputs
        x4 = self.flatten(x3)
        x5 = self.postprocessor(x4)

        return self.compute_logits(x5)

    def build(self, input_shape):
        super(BirdNet, self).build(input_shape)

    def get_config(self):
        return {"hidden_units":self.hidden_units,
                "lstm_hidden_units": self.lstm_hidden_units,
                "dropout_rate":self.dropout_rate}

