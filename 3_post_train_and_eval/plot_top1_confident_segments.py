#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:45:58 (ywatanabe)"


import sys

sys.path.append("eeg_dementia_classification")
from eeg_dementia_classification import utils
from models.MNet.PretrainedEnsembleMnet_1000 import PretrainedEnsembleMnet_1000
from utils import load_data_all
import skimage
import torch
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


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


def forward(data_all, ensemble_model):
    X, T, S = [], [], []
    for ii in range(len(data_all)):
        cropped = _crop_a_signal(data_all.iloc[ii]["eeg"], ws_pts=1000)
        X.append(cropped)
        T.append([data_all.iloc[ii]["disease_type"] for _ in range(len(cropped))])
        S.append([data_all.iloc[ii]["subject"] for _ in range(len(cropped))])
    X = np.vstack(X).transpose(0,2,1)
    T = np.hstack(T)
    S = np.hstack(S)

    preds = []

    bs = 64
    n_batches = len(X) // bs
    Ab = torch.rand(
        bs,
    ).cuda()
    Sb = torch.rand(
        bs,
    ).cuda()
    Mb = torch.rand(
        bs,
    ).cuda()
    for i_batch in range(n_batches + 1):
        start = i_batch * bs
        end = (i_batch + 1) * bs
        Xb = torch.tensor(X[start:end]).float().cuda()
        yb, _ = ensemble_model(
            Xb, Ab[: len(Xb)], Sb[: len(Xb)], Mb[: len(Xb)]
        )
        preds.append(yb.detach().cpu().numpy())

    preds = np.vstack(preds)

    return X, T, S, preds


def get_the_topk_TP_data(i_disease, X, T, S, y, k=30):
    indi_TP = (pred_class == i_disease) * (T == i_disease).numpy()
    X_TP = X[indi_TP]
    T_TP = T[indi_TP]
    S_TP = S[indi_TP]
    yy_TP = y[indi_TP].max(axis=-1)
    indi_sorted = np.argsort(yy_TP)
    X_TP_sorted = X_TP[indi_sorted]
    T_TP_sorted = T_TP[indi_sorted]
    S_TP_sorted = S_TP[indi_sorted]
    yy_TP_sorted = yy_TP[indi_sorted]
    return X_TP_sorted[-k:], T_TP_sorted[-k:], S_TP_sorted[-k:], yy_TP_sorted[-k:]

if __name__ == "__main__":
    import mngs

    # # Parameters
    DISEASE_TYPES = ["HV", "AD", "DLB", "NPH"]
    BIDS_DICT = {
        "Osaka": "BIDS_dataset_Osk_v5.3",
        "Kochi":"BIDS_dataset_v1.1_Kochi",
        "Nissei":"BIDS_dataset_Nissei_v1.1",
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
    ensemble_model = PretrainedEnsembleMnet_1000(DISEASE_TYPES).cuda()
    ensemble_model.models = [mm.cuda() for mm in ensemble_model.models]

    # Gets predictions
    X, T, S, y = forward(data_all, ensemble_model)
    T = torch.tensor([{"HV":0, "AD":1, "DLB":2, "NPH":3}[t] for t in T], dtype=int)
    pred_class = y.argmax(axis=-1)        

    # Gets top-10 high posterior segments of TP by each disease type
    k = 1
    X_topk_HV, T_topk_HV, S_topk_HV, yy_topk_HV = get_the_topk_TP_data(0, X, T, S, y, k=k)
    X_topk_AD, T_topk_AD, S_topk_AD, yy_topk_AD = get_the_topk_TP_data(1, X, T, S, y, k=k)
    X_topk_DLB, T_topk_DLB, S_topk_DLB, yy_topk_DLB = get_the_topk_TP_data(2, X, T, S, y, k=k)
    X_topk_NPH, T_topk_NPH, S_topk_NPH, yy_topk_NPH = get_the_topk_TP_data(3, X, T, S, y, k=k)
    print(len(X_topk_HV), len(X_topk_AD), len(X_topk_DLB), len(X_topk_NPH))

    montage = \
    [
        ["FP1 - A1"],
        ["F3 - A1"],
        ["C3 - A1"],
        ["P3 - A1"],
        ["O1 - A1"],
        ["FP2 - A2"],
        ["F4 - A2"],
        ["C4 - A2"],
        ["P4 - A2"],
        ["O2 - A2"],
        ["F7 - A1"],
        ["T7 - A1"],
        ["P7 - A1"],
        ["F8 - A2"],
        ["T8 - A2"],
        ["P8 - A2"],
        ["Fz - A1"],
        ["Cz - A1"],
        ["Pz - A1"]
    ]

    for axis in ["off", "on"]:
        fig, axes = plt.subplots(ncols=4, nrows=19, sharex=True, sharey=True)
        for i_XX, XX in enumerate([X_topk_HV.squeeze().T,
                        X_topk_AD.squeeze().T,
                        X_topk_DLB.squeeze().T,
                        X_topk_NPH.squeeze().T,
                        ]):
            XX = (XX - XX.mean()) / XX.std()
            for i_mm, mm in enumerate(montage):
                ax = axes[i_mm, i_XX]
                if i_mm == 0:
                    ax.set_title(["HV", "AD", "DLB", "iNPH"][i_XX])
                if i_XX == 0:
                    ax.set_ylabel(montage[i_mm])
                ax.plot(XX[:, i_mm], linewidth=.5, color="black")
                ax.axis(axis)
        # plt.show()
        mngs.io.save(fig, f"./results/figs/line/representative_traces/axis_{axis}.tiff")
