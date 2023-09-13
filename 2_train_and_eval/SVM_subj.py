#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:10:29 (ywatanabe)"

import sys
import mngs
sys.path.append("./eeg_dem_clf")
import utils
sys.path.append("./externals")
from dataloader.DataLoaderFiller import DataLoaderFiller
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mngs.ml.plt.aucs.roc_auc import roc_auc

def eval_bACC_and_ROC_AUC_single(dls, disease_types):
    bACCs = []
    ROC_AUCs = []
    for i_fold, dl in zip(range(5), dls):
        X_tes, T_tes, Sg_tes, Sl_tes, A_tes, G_tes, M_tes = dl.dataset.arrs_list

        # Band amplitude
        X_tes = mngs.dsp.rfft_bands(torch.tensor(X_tes), 500, normalize=True).reshape(len(X_tes), -1)

        # Model
        clf_seg = mngs.io.load(f"./results/models/SVC_fold_#{i_fold}_epoch_#49.pkl")
        pred_proba_tes = clf_seg.predict_proba(X_tes)

        pred_proba_sub_tes = []
        T_sub_tes = []
        for sub in np.unique(Sg_tes):
            indi = Sg_tes == sub
            pred_proba_sub_tes.append(pred_proba_tes[indi].mean(axis=0))
            T_sub_tes.append(np.unique(np.array(T_tes)[indi])[0])
        pred_proba_sub_tes = np.array(pred_proba_sub_tes)
            
        bACC = balanced_accuracy_score(pred_proba_sub_tes.argmax(axis=-1), np.array(T_sub_tes))
        bACCs.append(bACC)

        # ROC
        fig, metrics_dict = roc_auc(plt, T_sub_tes, pred_proba_sub_tes, DISEASE_TYPES)
        ROC_AUC = metrics_dict["roc_auc"]["macro"]
        ROC_AUCs.append(ROC_AUC)

    out = dict(
        bACC_mean=np.mean(bACCs).round(3),
        bACC_std=np.std(bACCs).round(3),
        ROC_AUC_mean=np.mean(ROC_AUCs).round(3),
        ROC_AUC_std=np.std(ROC_AUCs).round(3),
        )
    return out

class EnsembleSVM():
    def __init__(self,):
        self.models = [mngs.io.load(f"./results/models/SVC_fold_#{i_fold}_epoch_#49.pkl") for i_fold in range(5)]
    def __call__(self, x):
        # Band amplitude
        x = mngs.dsp.rfft_bands(torch.tensor(x), 500, normalize=True).reshape(len(x), -1)
        x = np.stack([model.predict_proba(x) for model in self.models], axis=0).mean(axis=0)
        return x

def eval_bACC_and_ROC_AUC_ensemble(dl, disease_types):
    X_tes, T_tes, Sg_tes, Sl_tes, A_tes, G_tes, M_tes = dl.dataset.arrs_list
    ensemble_model = EnsembleSVM()
    
    pred_proba_tes = ensemble_model(X_tes)

    pred_proba_sub_tes = []
    T_sub_tes = []
    for sub in np.unique(Sg_tes):
        print(sub)
        indi = np.array(Sg_tes) == sub
        pred_proba_sub_tes.append(pred_proba_tes[indi].mean(axis=0))
        assert len(np.unique(np.array(T_tes)[indi])) == 1
        T_sub_tes.append(np.unique(np.array(T_tes)[indi][0]))
    pred_proba_sub_tes = np.vstack(pred_proba_sub_tes)
    T_sub_tes = np.hstack(T_sub_tes)

    bACC = balanced_accuracy_score(pred_proba_sub_tes.argmax(axis=-1), np.array(T_sub_tes))
    fig, metrics_dict = roc_auc(plt, T_sub_tes, pred_proba_sub_tes, disease_types)

    out = dict(
        bACC=bACC.round(3),
        ROC_AUC=metrics_dict["roc_auc"]["macro"].round(3),
    )
    return out

if __name__ == "__main__":
    DISEASE_TYPES = ["HV", "AD", "DLB", "NPH"]

    # Osaka
    dlf = DataLoaderFiller(
        "./data/BIDS_Osaka",
        DISEASE_TYPES,
        drop_cMCI=True,
    )
    dls = []
    for i_fold in range(5):
        dlf.fill(i_fold, reset_fill_counter=True)
        dls.append(dlf.dl_tes)
    out_Osaka_single = eval_bACC_and_ROC_AUC_single(dls, DISEASE_TYPES)
    
    # Kochi
    dl_Kochi = utils.load_dl_kochi_or_nissei(
        "BIDS_dataset_v1.1_Kochi", ["HV", "AD", "DLB", "NPH"]
    )
    # Single
    dls = [dl_Kochi for _ in range(5)]
    out_Kochi_single = eval_bACC_and_ROC_AUC_single(dls, DISEASE_TYPES)    
    # Ensemble
    out_Kochi_ensemble = eval_bACC_and_ROC_AUC_ensemble(dl_Kochi, DISEASE_TYPES)

    # Nissei
    dl_Nissei = utils.load_dl_kochi_or_nissei(
        "BIDS_dataset_Nissei_v1.1", ["HV", "AD", "DLB", "NPH"]
    )
    dls = [dl_Nissei for _ in range(5)]
    out_Nissei_single = eval_bACC_and_ROC_AUC_single(dls, DISEASE_TYPES)        
    # Ensemble    
    out_Nissei_ensemble = eval_bACC_and_ROC_AUC_ensemble(dl_Nissei, DISEASE_TYPES)

    ## EOF
