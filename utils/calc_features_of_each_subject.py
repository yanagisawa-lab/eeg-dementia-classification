#!/usr/bin/env python3
# Time-stamp: "2021-11-25 06:46:25 (ylab)"

import numpy as np
import scipy.stats as st


def calc_features_of_each_subject(true_class, pred_proba, subjs, n_ftrs=1):
    # gets survived subjects indi
    subjs_uq = np.unique(subjs)
    indi = [subjs == su for su in subjs_uq]

    # X, feature engineering
    # pred_proba_subj = np.array(
    #     [pred_proba[indi[i]] for i in range(len(indi))],
    # )

    ftrs_subj = np.array(
        [
            extract_the_first_four_moments(pred_proba[indi[i]].T).T
            for i in range(len(indi))
        ]
    )  # RuntimeWarning
    ftrs_subj = ftrs_subj[:, :n_ftrs, :]  # fixme
    # IndexError: too many indices for array: array is 1-dimensional, but 3 were indexed

    # reshape
    ftrs_subj = ftrs_subj[..., :-1]
    ftrs_subj = ftrs_subj.reshape(len(ftrs_subj), -1)  # flatten

    ## subject's diagnosis label
    true_class_subj = np.array(
        [true_class[indi[i]].mean() for i in range(len(indi))]
    ).astype(int)

    subject_features_dict = {
        "ftrs": ftrs_subj,
        "true_class": true_class_subj,
        "subs": subjs_uq,
    }

    return subject_features_dict


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
