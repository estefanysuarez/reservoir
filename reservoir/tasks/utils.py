# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:50:52 2019

@author: Estefany Suarez
"""
import matplotlib.colors as mcolors
import numpy as np

#%% --------------------------------------------------------------------------------------------------------------------
# TASKS INPUTS/OUTPUTS
# ----------------------------------------------------------------------------------------------------------------------
def get_mem_cap_IO(input_nodes, n_nodes, time_steps = 4100, **kwargs):

    # same input sequence for multiple input nodes
    seq = np.random.uniform(-1, 1, (time_steps))

    inputs = np.zeros((len(seq), n_nodes))
    inputs[:,input_nodes] = np.repeat(seq[:,np.newaxis], len(input_nodes), axis=1)

    outputs = seq

    return inputs, outputs


def get_nonlin_cap_IO(input_nodes, n_nodes, time_steps = 4100, **kwargs):

    # same input sequence for multiple input nodes
    seq = np.random.uniform(-1, 1, (time_steps))

    inputs = np.zeros((len(seq), n_nodes))
    inputs[:,input_nodes] = np.repeat(seq[:,np.newaxis], len(input_nodes), axis=1)

    outputs = seq

    return inputs, outputs


def get_fcn_app_IO(input_nodes, n_nodes, time_steps = 4100, **kwargs):

    # same input sequence for multiple input nodes
    seq = np.random.uniform(-1, 1, (time_steps))

    inputs = np.zeros((len(seq), n_nodes))
    inputs[:,input_nodes] = np.repeat(seq[:,np.newaxis], len(input_nodes), axis=1)

    outputs = seq

    return inputs, outputs


def get_pattrn_rec_IO(input_nodes, n_nodes, n_patterns=100, time_lens=None, add_noisy_sample=True, **kwargs):

    if time_lens is None: time_lens=30*np.ones(n_patterns, dtype=np.int16)
    sections = [np.sum(time_lens[:idx]) for idx in range(1,len(time_lens))]
    patterns = np.split(np.random.randint(0, len(input_nodes), (np.sum(time_lens))),
                        sections,
                        axis=0)

    labels = list(range(n_patterns))


    def add_noisy_samples(patterns, labels, input_nodes, n_noisy_samples=49, n_jitters=None, **kwargs):
        noisy_patt = [*patterns]
        new_labels = [*labels]
        for _ in range(n_noisy_samples):
            new_patt = patterns
            for patt in new_patt:
                len_patt = len(patt)
                if n_jitters is None: n_jitters = int(0.1*len_patt)
                rnd_idx = np.random.randint(0, len_patt, (n_jitters))
                patt[rnd_idx] = np.random.randint(0, len(input_nodes), (n_jitters))

            noisy_patt.extend(new_patt)
            new_labels.extend(labels)

        return noisy_patt, new_labels

    if add_noisy_samples:
        samples, targets = add_noisy_samples(patterns=patterns,
                                             labels=labels,
                                             input_nodes=input_nodes,
                                             **kwargs)

    inputs = []
    outputs = []
    for idx_sample, sample in enumerate(samples):

        out_seq = -1*np.ones((len(sample), n_patterns), dtype=np.int16)
        out_seq[:, targets[idx_sample]] = 1

        in_seq = np.zeros((len(sample), n_nodes), dtype=np.int16)
        for idx_time, val in enumerate(sample):
            idx_input_node = input_nodes[val]
            in_seq[idx_time, idx_input_node] = 1

        inputs.append(in_seq)
        outputs.append(out_seq)

    inputs = np.vstack(inputs)
    outputs = np.vstack(outputs)

    return inputs, outputs


def create_inputs(task, input_nodes, n_nodes, **kwargs):

    if task == 'mem_cap':
        inputs, outputs = get_mem_cap_IO(input_nodes, n_nodes, **kwargs)

    if task == 'nonlin_cap':
        inputs, outputs = get_nonlin_cap_IO(input_nodes, n_nodes, **kwargs)

    if task == 'fcn_app':
        inputs, outputs = get_fcn_app_IO(input_nodes, n_nodes, **kwargs)

    if task == 'ptn_rec':
        inputs, outputs = get_pattrn_rec_IO(input_nodes, n_nodes, **kwargs)

    return inputs, outputs

#%% --------------------------------------------------------------------------------------------------------------------
# TASKS INPUTS/OUTPUTS
# ----------------------------------------------------------------------------------------------------------------------
def writeDict(dict, filename, sep):
    with open(filename, "a") as f:
        for k, v in dict.iteritems():
            f.write(k + sep + str(v) + '\n')
    f.close()


def readDict(filename, sep):

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    with open(filename, "r") as f:
        dict = {}

        for line in f:
            k, v = line.split(sep)

            print(is_number(v))

            if is_number(v): dict[k] = np.float(v)

            else: dict[k] = v

    f.close()
    return(dict)

#%% --------------------------------------------------------------------------------------------------------------------
# `PLOTTING`
# ----------------------------------------------------------------------------------------------------------------------
def array2cmap(X):

    N = X.shape[0]
    r = np.linspace(0., 1., N+1)
    r = np.sort(np.concatenate((r, r)))[1:-1]
    rd = np.concatenate([[X[i, 0], X[i, 0]] for i in range(N)])
    gr = np.concatenate([[X[i, 1], X[i, 1]] for i in range(N)])
    bl = np.concatenate([[X[i, 2], X[i, 2]] for i in range(N)])
    rd = tuple([(r[i], rd[i], rd[i]) for i in range(2 * N)])
    gr = tuple([(r[i], gr[i], gr[i]) for i in range(2 * N)])
    bl = tuple([(r[i], bl[i], bl[i]) for i in range(2 * N)])
    cdict = {'red': rd, 'green': gr, 'blue': bl}
    return mcolors.LinearSegmentedColormap('my_colormap', cdict, N)
