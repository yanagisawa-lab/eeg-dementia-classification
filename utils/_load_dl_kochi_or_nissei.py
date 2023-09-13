#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:57:43 (ywatanabe)"
import sys

import mngs
import pandas as pd

import numpy as np
sys.path.append("./externals")
from dataloader.load_BIDS import load_BIDS as load_BIDS_func
from dataloader.utils.divide_into_datasets import divide_into_datasets
from dataloader.utils.label_handlers import _expand_conc_classes_str
from dataloader.DataLoaderFiller import DataLoaderFiller

# load_dl_kochi_or_nissei -> load_data_all -> load_BIDS_func

def load_data_all(
    BIDS_DATASET_NAME,
    disease_types=["HV", "AD", "NPH", "DLB"],
    from_pkl=True,
    samp_rate=500,
    apply_notch_filter=True,
):
    load_params = mngs.io.load("./config/load_params.yaml")
    load_params["apply_notch_filter"] = apply_notch_filter

    load_params["num_folds"] = 2
    load_params["val_ratio_to_trn"] = 0
    load_params[
        "BIDS_root"
    ] = f"/storage/dataset/EEG/internal/EEG_DiagnosisFromRawSignals/{BIDS_DATASET_NAME}"
    load_params["random_state"] = 42


    data_all = load_BIDS_func(load_params)    

    ## to float32
    data_all["eeg"] = [eeg.astype(np.float32) for eeg in data_all["eeg"]]

    ## drops not-labeled? data
    load_params["conc_classes"] = disease_types  # e.g., ["AD", "DLB", "HV", "NPH"]
    expanded_classes = np.hstack(_expand_conc_classes_str(disease_types))

    indi_to_use = (
        pd.concat([data_all["disease_type"] == ec for ec in expanded_classes], axis=1)
        .sum(axis=1)
        .astype(bool)
    )

    data_not_used = data_all.loc[~indi_to_use]
    data_all = data_all.loc[indi_to_use]

    return load_params, data_all, data_not_used


def load_dl_kochi_or_nissei(
    BIDS_DATASET_NAME,
    disease_types,
    from_pkl=True,
    Dementia_or_MCI="Dementia",
):
    """
    BIDS_DATASET_NAME=f"BIDS_dataset_v1.1_Kochi"
    BIDS_DATASET_NAME=f"BIDS_dataset_Nissei_v1.1"
    disease_types=["HV", "AD", "DLB", "NPH"]
    """

    load_params, data_all, data_not_used = load_data_all(
        BIDS_DATASET_NAME,
        disease_types=disease_types,
        from_pkl=from_pkl,
    )
    # Nissei (35): HV: 0, AD: 17, DLB: 14, NPH: 4
    # Kochi (73):  HV: 3, AD: 45, DLB: 10, NPH: 15

    cv_dict = divide_into_datasets(data_all, load_params)
    data_uq = data_all[["subject", "disease_type", "cognitive_level"]].drop_duplicates()
    if Dementia_or_MCI == "Dementia":
        _data_uq = data_uq[data_uq["cognitive_level"] == "Dementia"]
    if Dementia_or_MCI == "MCI":
        _data_uq = data_uq[(data_uq["cognitive_level"] == "cMCI")]
    data_uq = _data_uq[["subject", "disease_type"]]        
    
    subs = list(data_uq["subject"])
    labels = [cv_dict["conc_class_str_2_label_int_dict"][c] for c in list(data_uq["disease_type"])]
    crop_len = 1000
    subj_str2int_local = {s: i for i, s in enumerate(subs)}
    subj_str2int_global = subj_str2int_local
    dl = DataLoaderFiller.crop_without_overlap(
        data_all,
        subs,
        labels,
        crop_len,
        subj_str2int_local,
        subj_str2int_global,
        dtype_np=np.float32,
        num_workers=16,
    )
    return dl


if __name__ == "__main__":
    load_params, data_all, data_not_used = load_data_all(
        # "BIDS_dataset_Osk_v5.3",        
        # "BIDS_dataset_v1.1_Kochi",
        "BIDS_dataset_Nissei_v1.1",        
        disease_types=["HV", "AD", "DLB", "NPH"],
        from_pkl=False,
        apply_notch_filter=True,
    )

    dl_kochi = load_dl_kochi_or_nissei(
        "BIDS_dataset_v1.1_Kochi", ["HV", "AD", "DLB", "NPH"]
    )
    
    batch = next(iter(dl_kochi))
    Xb, Tb, Sgb, Slb = batch
    len(np.unique(dl_kochi.dataset.arrs_list[-2]))  # 73

    # dl_nissei = load_dl_kochi_or_nissei(
    #     "BIDS_dataset_Nissei_v1.1", ["HV", "AD", "DLB", "NPH"],
    #     Dementia_or_MCI="MCI",
    # )
    # batch = next(iter(dl_nissei))
    # Xb, Tb, Sgb, Slb, Ab, Gb, Mb = batch
    # len(np.unique(dl_nissei.dataset.arrs_list[-2]))  # 35
