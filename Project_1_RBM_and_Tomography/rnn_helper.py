import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class RNN():

    def __init__(self, n_vis: int, n_hdn: int):
        self.n_vis = n_vis
        self.n_hdn = n_hdn

    def model(self):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(None, self.n_vis)))
        model.add(layers.SimpleRNN(self.n_hdn))
        model.add(layers.Dense(self.n_vis))
        return model


if __name__ == '__main__':
    rnn = RNN(2, 6)
    model = rnn.model()
    model.summary()
    model.compile()
    print(model.predict([[[0,1]]]))

