#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:03:38 (ywatanabe)"
import sys

sys.path.append("./externals")
from dataloader.DataLoaderFiller import DataLoaderFiller
import skimage
import pandas as pd
import mngs
from natsort import natsorted
import numpy as np
import re
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pingouin as pg
import statsmodels.formula.api as smf

def cvt_section2seg(data_all, institution):
    def _cvt_section2seg(row):
        window_size_pts = 1000
        signal = row.eeg
        cropped = skimage.util.view_as_windows(
            signal,
            window_shape=(signal.shape[0], window_size_pts),
            step=window_size_pts,
        ).squeeze()
        df = pd.DataFrame({"eeg": [c for c in cropped]})
        df["subject"] = row.subject
        df["disease_type"] = row.disease_type
        df["cognitive_level"] = row.cognitive_level
        return df

    out_df = pd.concat(
        [_cvt_section2seg(row) for _, row in data_all.iterrows()]
    ).reset_index()
    del out_df["index"]
    out_df["institution"] = institution
    return out_df


def find_50_segments(data_all):
    subs_uq = list(data_all.subject.unique())
    all_indi_subs = []
    for sub in subs_uq:
        indi_sub = mngs.gen.search(sub, data_all.subject)[0]
        indi_sub = natsorted(np.random.permutation(indi_sub)[:50])
        all_indi_subs.append(indi_sub)
    all_indi_subs = np.hstack(all_indi_subs)
    return data_all.iloc[all_indi_subs]


def load_segments():
    df = []
    for BIDS_ROOT in ["./data/BIDS_Osaka", "./data/BIDS_Nissei", "./data/BIDS_Kochi"]:
        # Loads
        DISEASE_TYPES = ["HV", "AD", "DLB", "NPH"]
        dlf = DataLoaderFiller(BIDS_ROOT, DISEASE_TYPES, load_BIDS_from_pkl=False)

        # Crops EEG data
        institution = BIDS_ROOT.split("./data/BIDS_")[1]
        data_all = cvt_section2seg(dlf.data_all, institution)

        # Drops unnecessary rows
        data_all = find_50_segments(data_all)

        df.append(data_all)
    return pd.concat(df)

def add_PSD(df):
    PSDs = []
    for ii in range(len(df)):
        try:
            PSD = mngs.dsp.calc_psd(df.iloc[ii].eeg, samp_rate=500, normalize=True).mean(axis=0)
        except Exception as e:
            print(e)
            # too many indices for array: array is 1-dimensional, but 2 were indexed
            PSD = None
        PSDs.append(PSD)
    df["PSD"] = PSDs
    return df

def calc_mm_ci(df_inst, max_hz=75):
    df_inst = df_inst[~df_inst.PSD.isna()]
    PSD_inst = np.vstack(df_inst.PSD)
    mm_inst, ci_inst = PSD_inst.mean(axis=0), 1.96*PSD_inst.std(axis=0)/np.sqrt(len(PSD_inst))
    mm_inst, ci_inst = mm_inst[:max_hz*2], ci_inst[:max_hz*2]
    return mm_inst, ci_inst
    

if __name__ == "__main__":
    df_all = load_segments()
    df_all = mngs.io.load("./results/PSD/df_all_PSD.pkl")

    # take average of PSD per subject
    df_all_ave = []
    freqs = np.array(df_all.PSD.iloc[0].index).astype(float)
    for sub in df_all.subject.unique():
        _df_sub = df_all[df_all.subject == sub]
        ave_PSD = np.vstack(_df_sub.PSD).mean(axis=0)
        df_tmp = pd.DataFrame({
            "freq": freqs,
            "PSD": ave_PSD,
        })

        df_tmp["subject"] = sub
        df_tmp["disease_type"] = _df_sub.disease_type.iloc[0]
        df_tmp["institution"] = _df_sub.institution.iloc[0]
        df_all_ave.append(df_tmp)
    df_all_ave = pd.concat(df_all_ave)
    df_all_ave = df_all_ave[["PSD", "freq", "disease_type", "institution", "subject"]]

    df_all_ave["freq"] = df_all_ave["freq"].astype(str)
    df_all_ave.dropna(inplace=True)
    df_all_ave = df_all_ave.reset_index()
    del df_all_ave["index"]


    df_all_ave["PSD"] = df_all_ave["PSD"].astype(float)
    df_all_ave["freq"] = df_all_ave["freq"].astype(float)    
    df_all_ave.to_csv("./results/PSD/PSD_values_and_freq_disease_type_and_institution.csv")

    df = df_all_ave

    # Define frequency bands
    bands = {'delta': (0.5, 4),
             'theta': (4, 8),
             'low_alpha': (8, 10),
             'high_alpha': (10, 13),
             'beta': (13, 32),
             'gamma': (32, 75)}

    # Add a new column to the dataframe indicating the band
    df['band'] = pd.cut(df['freq'], bins=[0.5, 4, 8, 10, 13, 32, 75], labels=['delta', 'theta', 'low_alpha', 'high_alpha', 'beta', 'gamma'])


    # Specify the model
    df = df.dropna()    
    df = df.reset_index()
    df = df[df.disease_type != "HV"]

    # standardize all numeric columns
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    model = smf.mixedlm("PSD ~ C(band, Treatment(reference='delta')) * C(disease_type, Treatment(reference='AD')) * C(institution, Treatment(reference='Osaka'))", df, groups=df["subject"])        

    # Fit the model
    result = model.fit()

    # Print the summary
    # This is the first table
    df1 = pd.DataFrame(result.summary().tables[0])
    df1.to_csv('./results/PSD/mixed_lm_table1.csv')

    # This is the second table
    df2 = pd.DataFrame(result.summary().tables[1])
    df2.to_csv('./results/PSD/mixed_lm_table2.csv')

    # Saves the results
    mngs.io.save(result.summary(), "./results/PSD/mixedlm_results.csv")

    ## EOF
