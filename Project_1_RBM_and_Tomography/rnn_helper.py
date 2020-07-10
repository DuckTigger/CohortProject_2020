import tensorflow as tf
from tensorflow.keras import layers
import torch
import pandas as pd
import numpy as np
import pickle as pkl
import os

import H2_energy_calculator

BATCH_SIZE=10

class RNN:

    def __init__(self, n_vis: int, n_hdn: int, optimiser: tf.keras.optimizers.Optimizer, loss_fn: tf.keras.losses.Loss):
        self.n_vis = n_vis
        self.n_hdn = n_hdn
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.rnn = self.model()

    def model(self):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(BATCH_SIZE, self.n_vis)))
        model.add(layers.SimpleRNN(self.n_hdn, return_sequences=True))
        model.add(layers.Dense(self.n_vis))
        return model

    @staticmethod
    def linear(inp, weight, bias):
        output = tf.matmul(inp, tf.transpose(weight))
        output += bias
        return output

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            batch = tf.reshape(batch, (1, BATCH_SIZE, self.n_vis))
            sample = self.rnn(batch)
            loss = tf.reduce_mean(self.loss_fn(batch, sample))
        grads = tape.gradient(loss, self.rnn.trainable_weights)
        self.optimiser.apply_gradients(zip(grads, self.rnn.trainable_weights))
        return loss


class RNNHelper:

    def __init__(self, epochs: int = 15, n_samples: int = 1000, lr: float = 0.03, n_hdn: int = 6, verbose: bool = False,
                 loc: str = './'):
        self.epochs = epochs
        self.n_samples = n_samples
        self.lr = lr
        self.n_hdn = n_hdn
        self.v = verbose
        self.loc = loc
        self.result_path = os.path.join(self.loc, 'rnn_results')
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimiser = tf.keras.optimizers.Adam(self.lr)

    @staticmethod
    def return_coeffs_samples():
        loc = './H2_data/'
        coeffs = np.loadtxt(os.path.join(loc, 'H2_coefficients.txt'))
        samples = []
        true_energies = []
        for c in coeffs:
            r = c[0]
            np_data = np.loadtxt(os.path.join(loc, 'R_{}_samples.txt'.format(str(r))))
            true_energy = H2_energy_calculator.energy_from_freq(torch.from_numpy(np_data), c)
            true_energies.append(true_energy)
            train_len, test_len = int(0.7 * len(np_data)), int(0.3 * len(np_data))
            sample = tf.data.Dataset.from_tensor_slices(np_data)
            train, test = sample.take(train_len), sample.take(test_len)
            train, test = train.batch(BATCH_SIZE), test.batch(BATCH_SIZE)
            samples.append((train, test))
        return coeffs, samples, true_energies

    def create_and_train_rnn(self, coeff, train, test, true_energy):
        results = pd.DataFrame()
        r = coeff[0]
        if self.v:
            print('\n\nR = {}'.format(r))
        rnn = RNN(self.n_vis, self.n_hdn, self.optimiser, self.loss_fn)
        model = rnn.rnn
        for e in range(self.epochs):
            if self.v:
                print('Start of epoch {}'.format(e))
            for step, x_batch in enumerate(train):
                loss = rnn.train_step(x_batch)
                if step % 100 == 0:
                    energies = []
                    acc = []
                    for init in test:
                        test = tf.reshape(init, (1, BATCH_SIZE, self.n_vis))
                        predictions = model.predict_on_batch(test)
                        clip_pred = tf.clip_by_value(predictions, 0, 1)
                        torch_pred = torch.from_numpy(clip_pred.numpy())
                        energy = H2_energy_calculator.energy_from_freq(torch_pred, coeff)
                        energies.append(energy)
                        accuracy = 1 - np.abs(np.abs(energy - true_energy) / true_energy)
                        acc.append(accuracy)
                    step_energy = tf.reduce_mean(energies)
                    step_accuracy = tf.reduce_mean(acc)
                    if self.v:
                        print('Step: {}, loss: {}\nEnergy: {}\nAccuracy: {}'.format(step, loss,
                                                                                    step_energy,
                                                                                    step_accuracy))
                    results.append({'epoch': e, 'step': step,
                                    'loss': loss, 'energy': step_energy, 'accuracy': step_accuracy},
                                   ignore_index=True)
            with open(os.path.join(self.result_path, 'R_{}.pkl'.format(r)), 'wb') as f:
                pkl.dump(results, f)

        if 'andrew' in os.getcwd():
            model.save('/home/andrew/Documents/PhD/CDLQuantum/saved_models/rnn_r_{}'.format(r))

    def iterate_over_r(self):
        coeffs, samples, true_energies = self.return_coeffs_samples()
        self.n_vis = samples[0][0].element_spec.shape[-1]
        for coeff, (train, test), true_energy in zip(coeffs, samples, true_energies):
            self.create_and_train_rnn(coeff, train, test, true_energy)


if __name__ == '__main__':
    os.chdir("/home/andrew/Documents/PhD/CDLQuantum/repository/Project_1_RBM_and_Tomography")
    rnn = RNNHelper()
    rnn.iterate_over_r()

