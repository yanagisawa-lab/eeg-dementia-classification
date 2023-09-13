#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:35:05 (ywatanabe)"

import pandas as pd
from glob import glob
import numpy as np
import mngs


def extract_bacc(ldir):
    stopped_str = ldir.split("/")[-2]
    stopped_bands = stopped_str.split("_stopped")[0].split("-")
    
    lpath = ldir + "dMCI+Dementia/balanced_acc.csv"
    bacc = mngs.io.load(lpath)["balanced_acc"].iloc[0]

    sr = pd.Series(index=stopped_bands+["bACC"],
                   data=np.array([True for _ in range(len(stopped_bands))]+[bacc]).T,
                   name=stopped_str,
                   )    
    return sr


# Preparation
LDIR = "./2_train_and_eval/MNet_1000_subj/HV_vs_AD_vs_DLB_vs_NPH/_MNet_1000_WindowSize-2.0-sec_MaxEpochs_50_no_sig_False_no_age_True_no_sex_True_no_MMSE_True_2022-0918-2017/subj-level/band_masking/"
LDIRs = glob(LDIR + "*")

df = pd.DataFrame(columns=["delta", "theta", "lalpha", "halpha", "beta", "gamma", "bACC"])

# Loads to a df
for ldir in LDIRs:
    ldir += "/"
    sr = extract_bacc(ldir)
    df = df.append(sr)

del df[""] # None

df = df.replace({1: False}).fillna(True)
df = df.sort_values(by=["delta", "theta", "lalpha", "halpha", "beta", "gamma"]).reset_index()

mngs.io.save(df, "./results/bands_stopping_exp.csv")
