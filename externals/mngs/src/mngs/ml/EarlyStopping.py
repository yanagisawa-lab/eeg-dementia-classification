#!/usr/bin/env python3
# Time-stamp: "2021-12-18 16:58:11 (ywatanabe)"

import numpy as np
import mngs
import os


class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation score improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.delta = delta

        self.spaths_and_models_dict = {None: None, None: None}

        self.i_epoch = 0
        self.i_global = 0
        self.trace_func = trace_func

    def __call__(
        self, val_score, spaths_and_models_dict, i_epoch, i_global
    ):

        score = -val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(
                val_score, spaths_and_models_dict, i_epoch, i_global
            )

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"\nEarlyStopping counter: {self.counter} out of {self.patience}\n"
            )
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(
                val_score, spaths_and_models_dict, i_epoch, i_global
            )
            self.counter = 0

    def save_checkpoint(
        self, val_score, spaths_and_models_dict, i_epoch, i_global
    ):
        """Saves model when validation score decrease."""
        if self.verbose:
            self.trace_func(
                f"\nValidation score decreased ({self.val_score_min:.6f} --> "
                f"{val_score:.6f}).  Saving model ...\n"
            )

        for spath, model in spaths_and_models_dict.items():
            try:
                mngs.io.save(model.state_dict(), spath) # torch
            except Exception as e:
                print(e)
                mngs.io.save(model, spath)

        self.i_epoch = i_epoch
        self.i_global = i_global

        ## Update file
        try:
            # rm one-step-old spath
            for spath in list(self.spaths_and_models_dict.keys()):
                os.remove(spath)
                print(f"\nRemoved {spath}\n")
        except Exception as e:
            print(e)

        self.spaths_and_models_dict = spaths_and_models_dict

        self.val_score_min = val_score
