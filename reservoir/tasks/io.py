# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:50:52 2019

@author: Estefany Suarez
"""
import random
random.seed(370)

import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import sweep_poly

#%% --------------------------------------------------------------------------------------------------------------------
# TASKS INPUTS/OUTPUTS
# ----------------------------------------------------------------------------------------------------------------------
def get_io_data(task, task_ref, **kwargs):

    if task == 'sgnl_recon':
        inputs, outputs = get_sgnl_recon_IO(task_ref, **kwargs)

    if task == 'pttn_recog':
        inputs, outputs = get_pattrn_rec_IO(task_ref, **kwargs)

    return inputs, outputs


def get_sgnl_recon_IO(task_ref='T1', time_len=1000, step_len=20, bias=0.5, n_repeats=3, **kwargs):

    if task_ref == 'T1':
        input_train = np.random.uniform(-1, 1, (time_len))[:, np.newaxis]
        input_test  = np.random.uniform(-1, 1, (time_len))[:, np.newaxis]

    if task_ref == 'T2':

        def get_seq(step_duration):
            pos_step = (bias + np.random.uniform(-0.5, 0.5, step_len))#[:, np.newaxis]
            neg_step = (-bias + np.random.uniform(-0.5, 0.5, step_len))#[:, np.newaxis]
            return np.hstack([pos_step, np.zeros(step_len), \
                              neg_step, np.zeros(step_len)]
                             )

        input_train = np.hstack([get_seq(step_len) for _ in range(n_repeats)])
        input_test  = np.hstack([get_seq(step_len) for _ in range(n_repeats)])

    return (input_train, input_test), (input_train.copy(), input_test.copy())


def get_pattrn_rec_IO(task_ref='T2', n_input_nodes=10, gain=3, n_patterns=10, n_repeats=100, time_len=50, **kwargs):

    if task_ref == 'T1': #spike trains
        "Generates random patterns of single spikes"
        # create original patterns and labels
        time_lens=time_len*np.ones(n_patterns, dtype=np.int16)
        sections = [np.sum(time_lens[:idx]) for idx in range(1,len(time_lens))]
        patterns = np.split(np.random.randint(0, n_input_nodes, (np.sum(time_lens))),
                            sections,
                            axis=0)

        labels = list(range(n_patterns))

        # add noise to original patterns
        def add_noisy_samples(patterns, labels, n_input_nodes, n_repeats, n_jitters=None, **kwargs):
            noisy_patt = [*patterns]
            new_labels = [*labels]

            for _ in range(n_repeats-1):
                new_repeat = []
                for patt in patterns:
                    tmp_pattern = patt.copy()
                    if n_jitters is None: n_jitters = int(0.1*len(patt))
                    rnd_idx = np.random.choice(len(patt), n_jitters, replace=False)
                    tmp_pattern[rnd_idx] = np.random.randint(0, n_input_nodes, n_jitters)
                    new_repeat.append(tmp_pattern)

                noisy_patt.extend(new_repeat)
                new_labels.extend(labels)

            return noisy_patt, new_labels

        samples, targets = add_noisy_samples(patterns=patterns,
                                             labels=labels,
                                             n_input_nodes=n_input_nodes,
                                             n_repeats=n_repeats,
                                             **kwargs)

        # create inputs and outputs
        inputs = []
        outputs = []
        for sample, target in zip(samples, targets):

            out_seq = -1*np.ones((len(sample), n_patterns), dtype=np.int16)
            out_seq[:, target] = 1

            in_seq = np.zeros((len(sample), n_input_nodes))#, dtype=np.int16)
            for idx_time, val in enumerate(sample):
                # idx_input_node = input_nodes[val]
                in_seq[idx_time, val] = gain


            inputs.append(in_seq)
            outputs.append(out_seq)

        # split training/test sets
        train_frac = 0.5
        n_train_samples = int(round(n_repeats * train_frac)) * n_patterns

        x_train = np.vstack(inputs[:n_train_samples])
        x_test  = np.vstack(inputs[n_train_samples:])

        y_train = np.vstack(outputs[:n_train_samples])
        y_test  = np.vstack(outputs[n_train_samples:])

    if task_ref == 'T2':
        "Generates noisy, random sinusoidal patterns with variable frequency"

        coeffs = [np.random.uniform(-2, 2, size=4) for _ in range(n_patterns)]
        t = np.linspace(0, 10, time_len)

        patterns = []
        labels = []
        for _ in range(n_repeats):
            for label in range(n_patterns):
                coeff = coeffs[label]
                poly = np.poly1d([coeff[0], coeff[1], coeff[2], coeff[3]])
                w = gain*sweep_poly(t, poly)
                w += np.random.normal(0, 0.1, len(w))

                labels.append(label)
                patterns.append(w)

        data = list(zip(patterns, labels))

        # split training/test sets
        train_frac = 0.5
        n_train_samples = int(train_frac*n_repeats)*n_patterns

        train_data = data[:n_train_samples]
        test_data = data[n_train_samples:]

        # shuffle data
        random.shuffle(train_data)
        random.shuffle(test_data)

        train_patterns, train_labels = zip(*train_data)
        test_patterns, test_labels = zip(*test_data)

        x_train = []
        y_train = []
        for pattern, label in zip(train_patterns, train_labels):
            new_label = -1*np.ones((len(pattern), n_patterns), dtype=np.int16)
            new_label[:, label] = 1

            x_train.append(pattern[:,np.newaxis])
            y_train.append(new_label)

        x_test = []
        y_test = []
        for pattern, label in zip(test_patterns, test_labels):
            new_label = -1*np.ones((len(pattern), n_patterns), dtype=np.int16)
            new_label[:, label] = 1

            x_test.append(pattern[:,np.newaxis])
            y_test.append(new_label)

        # concatenate data
        x_train = np.vstack(x_train)
        y_train = np.vstack(y_train)

        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test)

    return (x_train, x_test), (y_train, y_test)
