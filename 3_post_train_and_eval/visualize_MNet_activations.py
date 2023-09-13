#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:52:12 (ywatanabe)"


import sys

sys.path.append("eeg_dementia_classification")
from eeg_dementia_classification import utils

from models.MNet.PretrainedEnsembleMNet_1000 import PretrainedEnsembleMNet_1000

from utils import load_data_all
import skimage
import torch
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import mngs
import seaborn as sns
from scipy.special import softmax
from scipy.stats import ks_2samp


def _crop_a_signal(
    sig_arr_trial,
    rand_init_value=0,
    ws_pts=100,
    start_cut_pts=0,
    end_cut_pts=0,
):
    """
    sig_arr_trial: (n_chs, 1600)
    viewed       : (15, 100, n_chs)

    Example:
        n_chs = 6
        seq_len = 1600
        sig_arr_trial = np.random.rand(n_chs, seq_len)
        _crop_a_signal(sig_arr_trial)
    """

    # to 2D
    if sig_arr_trial.ndim == 1:
        sig_arr_trial = sig_arr_trial[np.newaxis, ...]

    n_chs = len(sig_arr_trial)

    # slides the signal
    rand_init_value = rand_init_value % ws_pts
    sig_arr_trial = sig_arr_trial[
        :, start_cut_pts : sig_arr_trial.shape[-1] - end_cut_pts
    ]
    sig_arr_trial = sig_arr_trial[:, rand_init_value:]

    # crops the signal
    viewed = skimage.util.view_as_windows(
        sig_arr_trial.T,
        window_shape=(ws_pts, n_chs),
        step=ws_pts,
    )  # .squeeze()

    # to 3D
    if viewed.ndim == 4:
        viewed = viewed.squeeze(axis=1)
    # to 3D
    if viewed.ndim == 2:
        viewed = viewed[..., np.newaxis]
    # to 3D
    if viewed.ndim == 1:
        viewed = viewed[np.newaxis, ..., np.newaxis]

    return viewed


def get_uq_mean(T, S, y, acts_fc1, acts_fc2, acts_fc3):
    T_uq, S_uq, y_mean, acts_fc1_mean, acts_fc2_mean, acts_fc3_mean = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for sub in np.unique(S):
        T_sub = T[S == sub]
        y_sub = y[S == sub]
        acts_fc1_sub = acts_fc1[S == sub]
        acts_fc2_sub = acts_fc2[S == sub]
        acts_fc3_sub = acts_fc3[S == sub]
        T_uq.append(T_sub[0])
        S_uq.append(sub)
        y_mean.append(y_sub.mean(axis=0))
        acts_fc1_mean.append(acts_fc1_sub.mean(axis=0))
        acts_fc2_mean.append(acts_fc2_sub.mean(axis=0))
        acts_fc3_mean.append(acts_fc3_sub.mean(axis=0))
    T_uq = np.hstack(T_uq)
    S_uq = np.hstack(S_uq)
    y_mean = np.vstack(y_mean)
    acts_fc1_mean = np.vstack(acts_fc1_mean)
    acts_fc2_mean = np.vstack(acts_fc2_mean)
    acts_fc3_mean = np.vstack(acts_fc3_mean)
    return T_uq, S_uq, y_mean, acts_fc1_mean, acts_fc2_mean, acts_fc3_mean


def calc_corr_map(activations):
    # activations = softmax(ordered_acts_fc1_mean, axis=-1)
    corr_arr = np.nan * np.zeros([len(activations), len(activations)])
    for i in range(len(activations)):
        for j in range(len(activations)):
            corr_arr[i, j] = np.corrcoef(activations[i], activations[j])[0, 1]
    return corr_arr


def forward(data_all, model):
    X, T, S = [], [], []
    for ii in range(len(data_all)):
        cropped = _crop_a_signal(data_all.iloc[ii]["eeg"], ws_pts=1000)
        X.append(cropped)
        T.append([data_all.iloc[ii]["disease_type"] for _ in range(len(cropped))])
        S.append([data_all.iloc[ii]["subject"] for _ in range(len(cropped))])
    X = np.vstack(X)
    T = np.hstack(T)
    S = np.hstack(S)

    preds = []
    bs = 64
    n_batches = len(X) // bs
    Ab = torch.rand(
        bs,
    )
    Sb = torch.rand(
        bs,
    )
    Mb = torch.rand(
        bs,
    )
    for i_batch in range(n_batches + 1):
        start = i_batch * bs
        end = (i_batch + 1) * bs
        Xb = X[start:end]
        yb = model(
            torch.tensor(Xb).float(), Ab[: len(Xb)], Sb[: len(Xb)], Mb[: len(Xb)]
        )[0]
        preds.append(yb.detach().numpy())

    preds = np.vstack(preds)
    return T, S, preds


def brunner_munzel_test_for_correlation_map(corr_map):
    from itertools import combinations

    within_class_corrs = []
    between_class_corrs = []
    for ii, jj in combinations(range(len(corr_map)), 2):
        if ii >= jj:
            continue
        cls1 = labels[ii]
        cls2 = labels[jj]
        if cls1 == cls2:
            within_class_corrs.append(np.corrcoef(corr_map[ii], corr_map[jj])[0, 1])
        else:
            between_class_corrs.append(np.corrcoef(corr_map[ii], corr_map[jj])[0, 1])

    # print(np.mean(within_class_corrs))
    # print(np.mean(between_class_corrs))
    w, p, dof, eff = mngs.stats.brunner_munzel_test(
        within_class_corrs, between_class_corrs
    )
    return w, p, dof, eff


def get_triu(corr_map):
    _corr_map = corr_map.copy()
    n = _corr_map.shape[0]
    lower_triangle_indices = np.tril_indices(n, -1)
    _corr_map[lower_triangle_indices] = np.nan
    return _corr_map


if __name__ == "__main__":

    # # Parameters
    DISEASE_TYPES = ["HV", "AD", "DLB", "NPH"]
    BIDS_DICT = {
        "Osaka": "BIDS_dataset_Osk_v5.3",
        "Kochi": "BIDS_dataset_v1.1_Kochi",
        "Nissei": "BIDS_dataset_Nissei_v1.1",
    }

    # Loads
    _, data_all_Kochi, _ = load_data_all(
        BIDS_DICT["Kochi"],
        disease_types=["HV", "AD", "DLB", "NPH"],
        from_pkl=False,
        apply_notch_filter=True,
    )
    _, data_all_Nissei, _ = load_data_all(
        BIDS_DICT["Nissei"],
        disease_types=["HV", "AD", "DLB", "NPH"],
        from_pkl=False,
        apply_notch_filter=True,
    )

    data_all = pd.concat([data_all_Kochi, data_all_Nissei])

    # Model
    model = PretrainedEnsembleMNet_1000(DISEASE_TYPES, save_activations=True)

    # Gets predictions
    T, S, y = forward(data_all, model)
    acts_fc1 = np.array(
        [np.vstack(model.models[ii].activation_fc1) for ii in range(5)]
    ).mean(axis=0)
    acts_fc2 = np.array(
        [np.vstack(model.models[ii].activation_fc2) for ii in range(5)]
    ).mean(axis=0)
    acts_fc3 = np.array(
        [np.vstack(model.models[ii].activation_fc3) for ii in range(5)]
    ).mean(axis=0)
    T_uq, S_uq, y_mean, acts_fc1_mean, acts_fc2_mean, acts_fc3_mean = get_uq_mean(
        T, S, y, acts_fc1, acts_fc2, acts_fc3
    )

    sorted_indi_HV = np.argsort(-y_mean[T_uq == "HV"][:, 0])
    sorted_indi_AD = np.argsort(-y_mean[T_uq == "AD"][:, 1])
    sorted_indi_DLB = np.argsort(-y_mean[T_uq == "DLB"][:, 2])
    sorted_indi_NPH = np.argsort(-y_mean[T_uq == "NPH"][:, 3])

    ordered_acts_fc1_mean = np.vstack(
        [
            acts_fc1_mean[T_uq == "HV"][sorted_indi_HV],
            acts_fc1_mean[T_uq == "AD"][sorted_indi_AD],
            acts_fc1_mean[T_uq == "DLB"][sorted_indi_DLB],
            acts_fc1_mean[T_uq == "NPH"][sorted_indi_NPH],
        ]
    )

    ordered_acts_fc2_mean = np.vstack(
        [
            acts_fc2_mean[T_uq == "HV"][sorted_indi_HV],
            acts_fc2_mean[T_uq == "AD"][sorted_indi_AD],
            acts_fc2_mean[T_uq == "DLB"][sorted_indi_DLB],
            acts_fc2_mean[T_uq == "NPH"][sorted_indi_NPH],
        ]
    )

    ordered_acts_fc3_mean = np.vstack(
        [
            acts_fc3_mean[T_uq == "HV"][sorted_indi_HV],
            acts_fc3_mean[T_uq == "AD"][sorted_indi_AD],
            acts_fc3_mean[T_uq == "DLB"][sorted_indi_DLB],
            acts_fc3_mean[T_uq == "NPH"][sorted_indi_NPH],
        ]
    )

    n_HV = (T_uq == "HV").sum()
    n_AD = (T_uq == "AD").sum()
    n_DLB = (T_uq == "DLB").sum()
    n_NPH = (T_uq == "NPH").sum()
    n_all = n_HV + n_AD + n_DLB + n_NPH
    print(n_all, n_HV, n_AD, n_DLB, n_NPH)

    labels = np.zeros(len(T_uq), dtype=int)
    labels[:n_HV] = 0
    labels[n_HV : n_HV + n_AD] = 1
    labels[n_HV + n_AD : n_HV + n_AD + n_DLB] = 2
    labels[n_HV + n_AD + n_DLB : n_HV + n_AD + n_DLB + n_NPH] = 3

    # Activation maps
    fig, axes = plt.subplots(ncols=4, sharey=True, figsize=(6.4 * 3, 4.8 * 3))
    axes[0].set_ylabel("Subject #")
    axes[0].set_xlabel("Disease type")
    sns.heatmap(labels[..., np.newaxis], ax=axes[0])
    sns.heatmap(softmax(ordered_acts_fc1_mean, axis=-1), ax=axes[1])
    axes[1].set_title("fc1")
    axes[1].set_xlabel("Node #")
    sns.heatmap(softmax(ordered_acts_fc2_mean, axis=-1), ax=axes[2])
    axes[2].set_title("fc2")
    axes[2].set_xlabel("Node #")
    sns.heatmap(softmax(ordered_acts_fc3_mean, axis=-1), ax=axes[3])
    axes[3].set_title("fc3")
    axes[3].set_xlabel("Node #")
    mngs.io.save(fig, "./results/figs/heatmap/activations.png")

    corr_map_fc1 = calc_corr_map(softmax(ordered_acts_fc1_mean, axis=-1))
    corr_map_fc2 = calc_corr_map(softmax(ordered_acts_fc2_mean, axis=-1))
    corr_map_fc3 = calc_corr_map(softmax(ordered_acts_fc3_mean, axis=-1))
    mngs.io.save(corr_map_fc1, "./results/figs/hist/corr_map/fc1.npy")
    mngs.io.save(corr_map_fc2, "./results/figs/hist/corr_map/fc2.npy")
    mngs.io.save(corr_map_fc3, "./results/figs/hist/corr_map/fc3.npy")

    """
    corr_map_fc1 = mngs.io.load("./results/figs/hist/corr_map/fc1.npy")
    corr_map_fc2 = mngs.io.load("./results/figs/hist/corr_map/fc2.npy")
    corr_map_fc3 = mngs.io.load("./results/figs/hist/corr_map/fc3.npy")    
    """

    print(brunner_munzel_test_for_correlation_map(corr_map_fc1))
    print(brunner_munzel_test_for_correlation_map(corr_map_fc2))
    print(brunner_munzel_test_for_correlation_map(corr_map_fc3))
    """
    (-7.055405242445422, 1.7841284005726266e-12, 17841.353914308387, 0.47274769580246373)
    (0.8758849317488225, 0.3811043217924823, 17935.69268036828, 0.5033867785958483)
    (1.3379178199021384, 0.18094001733303133, 18120.31940680489, 0.5051657230347484)
    """

    # Correlation maps
    fig, axes = plt.subplots(ncols=3, figsize=(6.4 * 2, 4.8 * 2), sharey=True)
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])  # where to put the colorbar
    vmin, vmax = -1, 1
    corr_map_fc1[100, 0] = -1  # add -1 align the color bar
    corr_map_fc2[100, 0] = -1
    corr_map_fc3[100, 0] = -1
    sns.heatmap(
        corr_map_fc1, ax=axes[0], vmin=vmin, vmax=vmax
    )  # , cbar=True, cbar_ax=cbar_ax)
    axes[0].set_aspect(1.0)
    sns.heatmap(corr_map_fc2, ax=axes[1], vmin=vmin, vmax=vmax)  # , cbar=False)
    axes[1].set_aspect(1.0)
    sns.heatmap(corr_map_fc3, ax=axes[2], vmin=vmin, vmax=vmax)  # , cbar=False)
    axes[2].set_aspect(1.0)
    mngs.io.save(fig, "./results/figs/heatmap/corr_map_of_activations.tiff")

    # Histogram of correlations
    nn_fc1, bins_fc1, _ = plt.hist(
        np.hstack(get_triu(corr_map_fc1)), bins=30, range=[-1, 1], alpha=0.3
    )
    bins_fc1 = [
        (bins_fc1[ii] + bins_fc1[ii + 1]) / 2 for ii in range(len(bins_fc1) - 1)
    ]
    nn_fc2, bins_fc2, _ = plt.hist(
        np.hstack(get_triu(corr_map_fc2)), bins=30, range=[-1, 1], alpha=0.3
    )
    bins_fc2 = [
        (bins_fc2[ii] + bins_fc2[ii + 1]) / 2 for ii in range(len(bins_fc2) - 1)
    ]
    nn_fc3, bins_fc3, _ = plt.hist(
        np.hstack(get_triu(corr_map_fc3)), bins=30, range=[-1, 1], alpha=0.3
    )
    bins_fc3 = [
        (bins_fc3[ii] + bins_fc3[ii + 1]) / 2 for ii in range(len(bins_fc3) - 1)
    ]

    df = pd.DataFrame(
        {
            "bins_fc1": bins_fc1,
            "count_fc1": nn_fc1,
            "bins_fc2": bins_fc2,
            "count_fc2": nn_fc2,
            "bins_fc3": bins_fc3,
            "count_fc3": nn_fc3,
        }
    )
    mngs.io.save(df, "./results/figs/hist/correlations_of_act_map/data.csv")

    ks_2samp(
        corr_map_fc1.reshape(-1),
        corr_map_fc2.reshape(-1),
    )
    # Out[15]: KstestResult(statistic=0.3988861816916116, pvalue=0.0)

    ks_2samp(
        corr_map_fc2.reshape(-1),
        corr_map_fc3.reshape(-1),
    )
    # Out[16]: KstestResult(statistic=0.21483999099117546, pvalue=0.0)

    ks_2samp(
        corr_map_fc1.reshape(-1),
        corr_map_fc3.reshape(-1),
    )
    # Out[17]: KstestResult(statistic=0.4222886509285232, pvalue=0.0)
