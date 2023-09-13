#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:09:35 (ywatanabe)"

import numpy as np

# from catboost import CatBoostClassifier, Pool
from sklearn.svm import SVC
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(".")
# from eeg_dem_clf.utils.DataLoaderFiller_PJ import init_dlf
sys.path.append("./externals")
from dataloader.DataLoaderFiller import DataLoaderFiller
from sklearn import preprocessing
from tqdm import tqdm
import mngs

# Fixes random seed
mngs.general.fix_seeds(seed=42, np=np, torch=torch)

# Configures matplotlib
mngs.plt.configure_mpl(plt)

# Determines save dir
def determine_save_dir(disease_types, MODEL_NAME, window_size_sec):
    sdir_root = mngs.general.mk_spath("")
    comparison = mngs.general.connect_strs(disease_types, filler="_vs_")
    sdir = (
        sdir_root + f"{os.getenv('USER')}/{comparison}/"
        f"_{MODEL_NAME}_WindowSize-{window_size_sec}-sec"
        f"_{mngs.general.gen_timestamp()}/seg-level/"
    )
    return sdir


def train_and_predict_SVC(dlf, i_fold):
    ## Model
    clf_seg = SVC(probability=True)

    for epoch in tqdm(range(50)):
        dlf.fill(i_fold)

        ## Get EEG, Target (= Diagnosis labels)
        X_tra, T_tra, S_tra, Sr_tra, _, _, _ = dlf.dl_tra.dataset.arrs_list
        X_val, T_val, S_val, Sr_val, _, _, _ = dlf.dl_val.dataset.arrs_list
        X_tes, T_tes, S_tes, Sr_tes, _, _, _ = dlf.dl_tes.dataset.arrs_list

        X_tra = X_tra[..., 500:-500]
        X_tra = np.vstack([X_tra, X_val])
        T_tra = np.hstack([T_tra, T_val])

        # Band amplitude
        X_tra = mngs.dsp.rfft_bands(torch.tensor(X_tra), 500, normalize=True).reshape(len(X_tra), -1)
        X_tes = mngs.dsp.rfft_bands(torch.tensor(X_tes), 500, normalize=True).reshape(len(X_tra), -1)

        ## List to array
        T_tra = np.array(T_tra)
        T_tes = np.array(T_tes)

        ## Training
        clf_seg.fit(X_tra, T_tra)

        ## Save the trained model
        mngs.io.save(
            clf_seg, f"./results/models/SVC_fold_#{i_fold}_epoch_#{epoch:02d}.pkl"
        )

        ## Prediction
        true_class_tes = np.array(T_tes)
        pred_proba_tes = clf_seg.predict_proba(X_tes)
        pred_class_tes = np.argmax(pred_proba_tes, axis=1)

        acc = np.mean(true_class_tes == pred_class_tes)
        print(f"fold#{i_fold}; ACC: {acc:.3f}")

    return true_class_tes, pred_proba_tes, pred_class_tes


def main(disease_types, window_size_sec):
    """
    disease_types = ["HV", "AD", "DLB", "NPH"]
    window_size_sec = 2
    """

    sdir_seg = determine_save_dir(disease_types, "SVC", window_size_sec)
    sys.stdout, sys.stderr = mngs.general.tee(sys, sdir_seg)  # log
    reporter_seg = mngs.ml.ClassificationReporter(sdir_seg)

    num_folds = mngs.io.load("./config/load_params.yaml")["num_folds"]
    for i_fold in range(num_folds):
        """
        i_fold = 0
        """
        print(f"\n {'-'*40} fold#{i_fold} starts. {'-'*40} \n")

        ## Initializes Dataloader
        dlf = DataLoaderFiller(
            "./data/BIDS_Osaka",
            args.disease_types,
            drop_cMCI=True,
        )

        # dlf = init_dlf(i_fold, disease_types, window_size_sec)
        dlf.fill(i_fold=i_fold)

        ## Training and Prediction
        true_class_tes, pred_proba_tes, pred_class_tes = train_and_predict_SVC(
            dlf, i_fold
        )

        ## Metrics
        reporter_seg.calc_metrics(
            true_class_tes,
            pred_class_tes,
            pred_proba_tes,
            labels=dlf.load_params["conc_classes"],
            i_fold=i_fold,
        )

    reporter_seg.summarize()

    reporter_seg.save(meta_dict={})


if __name__ == "__main__":
    import argparse
    import mngs

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-dts",
        "--disease_types",
        default=["HV", "AD", "DLB", "NPH"],
        nargs="*",
        help=" ",
    )
    ap.add_argument("-ws", "--window_size_sec", default=2, type=int, help=" ")
    args = ap.parse_args()

    main(args.disease_types, args.window_size_sec)
