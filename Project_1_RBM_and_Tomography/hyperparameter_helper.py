import numpy as np
import itertools as it
import time
import pandas as pd
import os
import pickle as pkl
import torch
from typing import Tuple, List, Iterable

from RBM_helper import RBM
import H2_energy_calculator


class H2HyperparameterSearch:

    def __init__(self, lr_range: Tuple[float, float, int], n_hdn_range: Tuple[int, int, int],
                 epochs: int = 500, n_samples: int = 1000, coeff: np.array = None,
                 training_data: torch.Tensor = None,
                 n_param_samples: int = 0, save_loc: str = './', verbose: bool = False):

        self.training_data = torch.from_numpy(np.loadtxt("H2_data/R_1.2_samples.txt")) if training_data is None \
            else training_data
        self.coeff = np.loadtxt("H2_data/H2_coefficients.txt")[20, :] if coeff is None else coeff
        self.true_energy = H2_energy_calculator.energy_from_freq(self.training_data, self.coeff)

        self.v = verbose
        if self.v:
            print('True energy: {}'.format(self.true_energy))
        self.epochs = epochs
        self.n_samples = n_samples
        lr_stepper = self.stepper(lr_range)
        n_hdn_stepper = self.stepper(n_hdn_range)
        self.grid = it.product(lr_stepper, n_hdn_stepper)
        if n_param_samples:
            self.grid = self.sample_k_params(n_param_samples, self.grid)
        self.save_loc = save_loc

    @staticmethod
    def sample_k_params(k: int, chain: Iterable):
        """
        Sampling randomly from a generator of unknown length, adapted from:
        https://ballsandbins.wordpress.com/2014/04/13/distributedparallel-reservoir-sampling/
        """
        pool = set()
        reservoir = []
        try:
            for i in range(k):
                reservoir.append(next(chain))
        except StopIteration:
            Warning("Sample size is larger than the size of the parameter grid")
        i = k
        try:
            while 1:
                j = np.random.randint(0, i + 1)
                i += 1
                if j < k:
                    reservoir[j] = next(chain)
                else:
                    next(chain)
        except StopIteration:
            pool.update(reservoir)
            return pool

    @staticmethod
    def stepper(params: Tuple):
        if params is None:
            return (None)
        start, stop, step = params[0], params[1], params[2]
        step_size = (stop - start) / step
        if isinstance(stop, int):
            steps = (int(start + (n * step_size)) for n in range(step))
        else:
            steps = (start + (n * step_size) for n in range(step))
        return steps

    def train_given_params(self, lr: float, n_hdn: int):
        if self.v:
            print('Params: lr = {}, n_hdn = {}'.format(lr, n_hdn))
        results = pd.DataFrame()
        n_vis = self.training_data.shape[1]
        rbm = RBM(n_vis, n_hdn)

        sampling_time = 0
        start = time.time()
        for e in range(1, self.epochs + 1):
            rbm.train(self.training_data, lr=lr)

            if e % 100 == 0:
                if self.v:
                    sampling_start = time.time()
                    print("\nEpoch: ", e)
                    print("Sampling the RBM...")

                    # For sampling the RBM, we need to do Gibbs sampling.
                    # Initialize the Gibbs sampling chain with init_state as defined below.
                    init_state = torch.zeros(self.n_samples, n_vis)
                    RBM_samples = rbm.draw_samples(15, init_state)

                    print("Done sampling. Calculating energy...")

                    energies = H2_energy_calculator.energy(RBM_samples, self.coeff, rbm.wavefunction)
                    print("Energy from RBM samples: ", energies.item())
                    sampling_end = time.time()
                    sampling_time += sampling_end - sampling_start

        end = time.time()
        time_taken = end - start
        time_taken -= sampling_time
        init_state = torch.zeros(self.n_samples, n_vis)
        rbm_samples = rbm.draw_samples(100, init_state)
        energies = H2_energy_calculator.energy(rbm_samples, self.coeff, rbm.wavefunction)
        accuracy = np.abs(np.abs(energies.item() - self.true_energy) / self.true_energy) * 100
        results = results.append({'n_hidden': n_hdn, 'epochs': self.epochs, 'n_samples': self.n_samples,
                                  'molecule': 'H2', 'final_energy': energies.item(), 'true_energy': self.true_energy,
                                  'accuracy': accuracy, 'time': time_taken}, ignore_index=True)
        if self.v:
            print('Training finished.\nFinal energy: {}\nAccuracy: {:.2f}%\nTime taken sampling: {}'.format(
                energies.item(), accuracy, sampling_time))
        return results

    def search_hyperparams(self):
        results = pd.DataFrame()
        path = os.path.join(self.save_loc, 'H2_hyperparam_search.pkl')
        for lr, n_hdn in self.grid:
            result = self.train_given_params(lr, n_hdn)
            results = results.append(result, ignore_index=True)
            with open(path, 'wb') as f:
                pkl.dump(results, f)
        return results
