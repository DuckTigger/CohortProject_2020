import numpy as np
import pandas as pd
import pickle as pkl
import os
import torch
import time
import H2_energy_calculator
from RBM_helper import RBM


class RBMOnH2Train:

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
            sample = torch.from_numpy(np.loadtxt(os.path.join(loc, 'R_{}_samples.txt'.format(str(r)))))
            samples.append(sample)
        return coeffs, samples

    def train_rbm(self, coeff: np.array, training_data: torch.Tensor):
        results = pd.DataFrame()
        true_energy = H2_energy_calculator.energy_from_freq(training_data, coeff)
        if self.v:
            print('r = {}'.format(coeff[0]))
            print('True Energy: {}'.format(true_energy))
        n_vis = training_data.shape[1]
        rbm = RBM(n_vis, self.n_hdn)
        start = time.time()
        for e in range(1, self.epochs + 1):
            rbm.train(training_data)

            # now generate samples and calculate the energy
            if e % 100 == 0:
                if self.v:
                    print("\nEpoch: ", e)
                    print("Sampling the RBM...")

                    # For sampling the RBM, we need to do Gibbs sampling.
                    # Initialize the Gibbs sampling chain with init_state as defined below.
                    init_state = torch.zeros(self.n_samples, n_vis)
                    RBM_samples = rbm.draw_samples(15, init_state)

                    print("Done sampling. Calculating energy...")

                    energies = H2_energy_calculator.energy(RBM_samples, coeff, rbm.wavefunction)
                    print("Energy from RBM samples: ", energies.item())
        end = time.time()
        time_taken = end - start
        init_state = torch.zeros(self.n_samples, n_vis)
        rbm_samples = rbm.draw_samples(100, init_state)
        energies = H2_energy_calculator.energy(rbm_samples, coeff, rbm.wavefunction)
        accuracy = np.abs(np.abs(energies.item() - true_energy) / true_energy) * 100
        results = results.append({'n_hidden': self.n_hdn, 'learning_rate': self.lr, 'epochs': self.epochs,
                                  'n_samples': self.n_samples, 'molecule': 'H2', 'final_energy': energies.item(),
                                  'true_energy': true_energy, 'accuracy': accuracy, 'time': time_taken},
                                 ignore_index=True)
        if self.v:
            print('Training finished.\nFinal energy: {}\nAccuracy: {:.2f}%\n\n\n'.format(
                energies.item(), accuracy))
        return results

    def train_for_all_r(self):
        results = pd.DataFrame()
        coeffs, train_data = self.return_coeffs_samples()
        for coeff, data in zip(coeffs, train_data):
            res = self.train_rbm(coeff, data)
            results = results.append(res, ignore_index=True)
            with open(os.path.join(self.loc, 'H2_R_results.pkl'), 'wb') as f:
                pkl.dump(results, f)
        return results
