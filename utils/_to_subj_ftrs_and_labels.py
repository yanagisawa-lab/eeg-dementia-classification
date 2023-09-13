#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-07-09 22:27:56 (ywatanabe)"

import numpy as np
import scipy.stats as st

# import kurtosis, skew


def extract_the_first_four_moments(x):
    ## The first four moments of the data provided
    _means = np.mean(x, axis=-1, keepdims=True)
    _stds = np.std(x, axis=-1, keepdims=True)
    _skews = st.skew(x, axis=-1)[..., np.newaxis]
    _kurts = st.kurtosis(x, axis=-1)[..., np.newaxis]
    moments = np.stack(
        [
            _means,
            _stds,
            _skews,
            _kurts,
        ],
        axis=-1,
    ).squeeze()
    return moments

def to_subj_ftrs_and_labels(true_class, pred_proba, subjs, n_ftrs=3):
    """
    Summarize segment-level pred_proba as subject-level features.
    """

    # gets survived subjects indi
    subjs_uq = np.unique(subjs)
    indi = [subjs == su for su in subjs_uq]

    # X, feature engineering
    ftrs_subj = np.array(
        [
            extract_the_first_four_moments(pred_proba[indi[i]].T).T
            for i in range(len(indi))
        ]
    )  # RuntimeWarning
    ftrs_subj = ftrs_subj[:, :n_ftrs, :]  # fixme
    # [n_subjs, n_ftrs, n_classes]

    # reshape
    ftrs_subj = ftrs_subj.reshape(len(ftrs_subj), -1)
    # ftrs_subj = ftrs_subj[..., :-1] # the last elements have no info
    # ftrs_subj = ftrs_subj.squeeze() # flatten

    ## subject's diagnosis label
    true_class_subj = np.array(
        [true_class[indi[i]].mean() for i in range(len(indi))]
    ).astype(int)
    return ftrs_subj, true_class_subj, subjs_uq
