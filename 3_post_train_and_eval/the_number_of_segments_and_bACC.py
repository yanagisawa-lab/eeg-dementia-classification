#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:51:06 (ywatanabe)"


import sys

sys.path.append("eeg_dementia_classification")
from eeg_dementia_classification import utils
from utils import load_data_all
import skimage
import torch
import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from gradcam import GradCAM
from sklearn.utils import shuffle
from sklearn.metrics import balanced_accuracy_score
import random
from tqdm import tqdm
    
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


def calc_bacc(n_use,
              S_more_than_300_segs,
              T_more_than_300_segs,
              y_more_than_300_segs,                  
              ):
    
    X_all, T_all, S_all, y_all = [], [], [], []
    for sub in np.unique(S_more_than_300_segs):
        indi = S_more_than_300_segs == sub

        T_sub = T_more_than_300_segs[indi]
        S_sub = S_more_than_300_segs[indi]
        y_sub = y_more_than_300_segs[indi]

        assert (T_sub == T_sub[0]).all()
        assert (S_sub == S_sub[0]).all()        
        T_all.append(T_sub[0])
        S_all.append(S_sub[0])

        y_sub_chosen = np.vstack(y_sub[np.random.permutation(np.arange(len(y_sub))).squeeze()[:n_use]])
        y_sub_mean = y_sub_chosen.mean(axis=0)
        y_all.append(y_sub_mean)

    y_all = np.vstack(y_all)
    pred_class_all = y_all.argmax(axis=-1)
    bacc = balanced_accuracy_score(np.hstack(T_all), pred_class_all)
    return bacc

if __name__ == "__main__":
    import mngs
    
    mngs.gen.fix_seeds(random=random)
    
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
    ensemble_model = EnsembleModel(DISEASE_TYPES).cuda()
    ensemble_model.models = [mm.cuda() for mm in ensemble_model.models]

    # Gets predictions
    X, T, S, y = forward(data_all, ensemble_model)
    T = torch.tensor([{"HV":0, "AD":1, "DLB":2, "NPH":3}[t] for t in T], dtype=int)
    pred_class = y.argmax(axis=-1)

    S_uq, ns = np.unique(S, return_counts=True)
    S_more_than_300_segs_uq = S_uq[ns > 300]


    indi_use = mngs.gen.search(list(S_more_than_300_segs_uq), list(S))[0]
    X_more_than_300_segs = X[indi_use]
    T_more_than_300_segs = T[indi_use]
    S_more_than_300_segs = S[indi_use]
    y_more_than_300_segs = y[indi_use]


    dd = mngs.gen.listed_dict()
    for n_use in tqdm([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]):
        for _ in range(100):
            dd[n_use].append(calc_bacc(n_use, S_more_than_300_segs, T_more_than_300_segs, y_more_than_300_segs))
        
    df = pd.DataFrame(dd)
    xx = df.columns
    mm = df.mean()
    ss = df.std()
    ci = 1.96*ss/np.sqrt(len(df))

    fig, ax = plt.subplots()
    ax = mngs.plt.ax_fill_between(ax, xx, mm, ci, "")
    ax.set_xlabel("# of 2-s segments")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Hospital K and N")
    mngs.io.save(fig, "./results/figs/line/the_number_of_segments_and_bACC/fig.png")    
    # plt.show()
    

    out_df = pd.DataFrame({
        "xx": xx,
        "under": mm - ci,
        "mean": mm,
        "upper": mm + ci,        
    })
    mngs.io.save(out_df, "./results/figs/line/the_number_of_segments_and_bACC/data.csv")
