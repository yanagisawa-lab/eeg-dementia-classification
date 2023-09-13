#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:39:01 (ywatanabe)"

import pandas as pd
from glob import glob
import numpy as np
import mngs


def extract_bacc(ldir):
    lpath = ldir + "dMCI+Dementia/balanced_acc.csv"
    bacc = mngs.io.load(lpath)["balanced_acc"].iloc[0]
    return bacc


# Preparation
LDIR = "./train/MNet_1000_subj/submission_2022_0919/HV_vs_AD_vs_DLB_vs_NPH/_MNet_1000_WindowSize-2.0-sec_MaxEpochs_50_no_sig_False_no_age_True_no_sex_True_no_MMSE_True_2022-0918-2017/subj-level/ch_masking_2022_0923/"
LDIRs = glob(LDIR + "*")


df = mngs.io.load("./results/montages_to_mask.csv", index_col="rand_index")
df = df.iloc[:len(LDIRs)]

df["bACC"] = None
for i_ldir, ldir in enumerate(LDIRs):
    try:
        df.loc[i_ldir, "bACC"] = extract_bacc(ldir + "/")
    except Exception as e:
        print(e)
        pass

# df["bACC"].isna().sum() # 4
drop_indi = df["bACC"].isna()
df = df[~drop_indi]



mngs.io.save(df, "./results/channels_stopping_exp.csv")
