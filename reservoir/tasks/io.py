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
def get_io_data(task, task_ref, n_nodes, input_nodes, **kwargs):

    def stack_seq(seq, n_nodes, input_nodes):

        train_seq, test_seq = seq

        input_train = np.zeros((len(train_seq), n_nodes))
        input_train[:,input_nodes] = np.repeat(train_seq[:,np.newaxis], len(input_nodes), axis=1)

        input_test = np.zeros((len(test_seq), n_nodes))
        input_test[:,input_nodes] = np.repeat(test_seq[:,np.newaxis], len(input_nodes), axis=1)

        return (input_train, input_test)

    if task == 'sgnl_recon':
        inputs, outputs = get_sgnl_recon_IO(task_ref, **kwargs)
        inputs = stack_seq(inputs, n_nodes, input_nodes)

    if task == 'pttn_recog':
        inputs, outputs = get_pattrn_rec_IO(task_ref, input_nodes, n_nodes, **kwargs)

    return inputs, outputs


def get_sgnl_recon_IO(task_ref = 'T1', time_len = 1000, step_len = 20, bias = 0.5, n_repeats=3, **kwargs):

    if task_ref == 'T1':
        input_train = np.random.uniform(-1, 1, (time_len))
        input_test  = np.random.uniform(-1, 1, (time_len))

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


def get_pattrn_rec_IO(task_ref, input_nodes, n_nodes, n_patterns=10, n_repeats=80, time_lens=None, **kwargs):

    if task_ref == 'T1': #spike trains

        # create patterns and labels
        if time_lens is None: time_lens=30*np.ones(n_patterns, dtype=np.int16)
        sections = [np.sum(time_lens[:idx]) for idx in range(1,len(time_lens))]
        patterns = np.split(np.random.randint(0, len(input_nodes), (np.sum(time_lens))),
                            sections,
                            axis=0)

        labels = list(range(n_patterns))

        # add noise to patterns
        def add_noisy_samples(patterns, labels, input_nodes, n_repeats, n_jitters=None, **kwargs):
            noisy_patt = [*patterns]
            new_labels = [*labels]
            for _ in range(n_repeats-1):
                new_patt = patterns
                for patt in new_patt:
                    len_patt = len(patt)
                    if n_jitters is None: n_jitters = int(0.1*len_patt)
                    rnd_idx = np.random.randint(0, len_patt, (n_jitters))
                    patt[rnd_idx] = np.random.randint(0, len(input_nodes), (n_jitters))

                noisy_patt.extend(new_patt)
                new_labels.extend(labels)

            return noisy_patt, new_labels

        samples, targets = add_noisy_samples(patterns=patterns,
                                             labels=labels,
                                             input_nodes=input_nodes,
                                             n_repeats=n_repeats,
                                             **kwargs)

        inputs = []
        outputs = []
        for sample, target in zip(samples, targets):

            out_seq = -1*np.ones((len(sample), n_patterns), dtype=np.int16)
            out_seq[:, target] = 1

            in_seq = np.zeros((len(sample), n_nodes), dtype=np.int16)
            for idx_time, val in enumerate(sample):
                idx_input_node = input_nodes[val]
                in_seq[idx_time, idx_input_node] = 1

            inputs.append(in_seq)
            outputs.append(out_seq)

        # split training/test sets
        train_frac = 0.5
        n_train_samples = int(round(n_repeats * train_frac)) * n_patterns
        n_test_samples  = int(round(n_repeats * (1 - train_frac))) * n_patterns

        x_train = np.vstack(inputs[:n_train_samples])
        x_test  = np.vstack(inputs[n_train_samples:])

        y_train = np.vstack(outputs[:n_train_samples])
        y_test  = np.vstack(outputs[n_train_samples:])

    return (x_train, x_test), (y_train, y_test)


#%% --------------------------------------------------------------------------------------------------------------------
# TASKS INPUTS/OUTPUTS
# ----------------------------------------------------------------------------------------------------------------------
# def writeDict(dict, filename, sep):
#     with open(filename, "a") as f:
#         for k, v in dict.iteritems():
#             f.write(k + sep + str(v) + '\n')
#     f.close()
#
#
# def readDict(filename, sep):
#
#     def is_number(s):
#         try:
#             float(s)
#             return True
#         except ValueError:
#             pass
#
#         try:
#             import unicodedata
#             unicodedata.numeric(s)
#             return True
#         except (TypeError, ValueError):
#             pass
#
#         return False
#
#     with open(filename, "r") as f:
#         dict = {}
#
#         for line in f:
#             k, v = line.split(sep)
#
#             print(is_number(v))
#
#             if is_number(v): dict[k] = np.float(v)
#
#             else: dict[k] = v
#
#     f.close()
#     return(dict)
