import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os


class RNN:

    def __init__(self, n_vis: int, n_hdn: int):
        self.n_vis = n_vis
        self.n_hdn = n_hdn

    def model(self):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(None, self.n_vis)))
        model.add(layers.Bidirectional(layers.SimpleRNN(self.n_hdn)))
        model.add(layers.Dense(self.n_vis))
        return model


class RNNHelper:

    def __init__(self, epochs: int = 500, n_samples: int = 1000, lr: float = 0.03, n_hdn: int = 6, verbose: bool = False,
                 loc: str = './'):
        self.epochs = epochs
        self.n_samples = n_samples
        self.lr = lr
        self.n_hdn = n_hdn
        self.v = verbose
        self.loc = loc

    @staticmethod
    def return_coeffs_samples():
        loc = './H2_data/'
        coeffs = np.loadtxt(os.path.join(loc, 'H2_coefficients.txt'))
        samples = []
        for c in coeffs:
            r = c[0]
            np_data = np.loadtxt(os.path.join(loc, 'R_{}_samples.txt'.format(str(r))))
            train_len, test_len = int(0.7 * len(np_data)), int(0.3 * len(np_data))
            sample = tf.data.Dataset.from_tensor_slices([np_data])
            train, test = sample.take(train_len), sample.take(test_len)
            train, test = train.padded_batch(10), test.padded_batch(10)
            samples.append((train, test))
        return coeffs, samples

    def iterate_over_r(self):
        coeffs, samples = self.return_coeffs_samples()
        n_vis = samples[0][0].element_spec.shape[-1]

        for coeff, (train, test) in zip(coeffs, samples):
            rnn = RNN(n_vis, self.n_hdn)
            model = rnn.model()
            model = self.train_model(model, train, test)
            # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            #               optimizer=tf.keras.optimizers.Adam(self.lr), )
            # history = model.fit(train, epochs=self.epochs,
            #                     validation_data=test, validation_steps=30)

    def train_model(self, model: tf.keras.Model, train: tf.data.Dataset, test: tf.data.Dataset):
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        optimiser = tf.keras.optimizers.Adam(self.lr)
        for e in range(self.epochs):
            for step, x_batch in enumerate(train):
                with tf.GradientTape() as tape:
                    sample = model(x_batch)
                    batch_expectation = tf.reduce_mean(x_batch, 1)
                    loss = loss_fn(batch_expectation, sample)
                grads = tape.gradient(loss, model.trainable_weights)
                optimiser.apply_gradients(zip(grads, model.trainable_weights))
        return model


if __name__ == '__main__':
    os.chdir("/home/andrew/Documents/PhD/CDLQuantum/repository/Project_1_RBM_and_Tomography")
    rnn = RNNHelper()
    rnn.iterate_over_r()

