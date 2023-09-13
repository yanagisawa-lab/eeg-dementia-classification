#!/usr/bin/env python

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg

import seaborn as sns

from functools import partial


import mngs
import utils

sys.path.append("./externals")
from dataloader.DataLoaderFiller import DataLoaderFiller


## Functions
def _calc_fft_amps(samp_rate, arr_2d):
    return mngs.dsp.calc_fft_amps(arr_2d, samp_rate)


def _calc_bands_mean(fft_out):
    def _inner_calc_band_mean(fft_out, low_hz, high_hz):
        columns = np.array([float(s) for s in list(fft_out.columns)])
        indi = (low_hz <= columns) & (columns < high_hz)
        return fft_out.iloc[:, indi].sum(axis=1)

    out_dict = {}
    for band_str, (low_hz, high_hz) in BANDS_LIM_HZ_DICT.items():
        out_dict[band_str] = pd.DataFrame(
            _inner_calc_band_mean(fft_out, low_hz, high_hz)
        ).set_index(pd.Index(montage))

    return pd.concat(out_dict, axis=1)


def rename_band_names(band_names_list):
    CVT_DICT = dict(
        delta="$\delta$",
        theta="$\\theta$",
        lalpha="$low\ \\alpha$",
        halpha="$high\ \\alpha$",
        beta="$\\beta$",
        gamma="$\\gamma$",
    )
    return [CVT_DICT[bn] for bn in band_names_list]


def extract_a_kind_of_value(column_name, dfs_simple):
    """
    column_name: "F", "p-unc", or "np2"
    """
    vals = np.nan * np.random.rand(19, 6)  # init
    val_df = dfs_simple[column_name].reset_index().set_index("Montage")

    MONTAGES = val_df.index.unique()  # will be organized as this order
    montage_wo_ref_uq = [mm.split("-")[0] for mm in MONTAGES]

    bands_uq = val_df["Freq. Band"].unique()

    for i_band, band in enumerate(bands_uq):
        vals[:, i_band] = val_df[val_df["Freq. Band"] == band].loc[MONTAGES][
            column_name
        ]

    return vals, montage_wo_ref_uq, bands_uq


def calc_bands_mean_of_a_dataset(BIDS_DATASET_NAME, disease_types, samp_rate):
    ## Loads
    BIDS_ROOT = (
        f"/storage/dataset/EEG/internal/EEG_DiagnosisFromRawSignals/"
        f"{BIDS_DATASET_NAME}"
    )

    dlf = DataLoaderFiller(BIDS_ROOT, disease_types, load_BIDS_from_pkl=False)
    data_all = dlf.data_all

    ## Calculate FFT Amps
    p_calc_fft_amps = partial(_calc_fft_amps, samp_rate)
    fft_amps = data_all["eeg"].apply(p_calc_fft_amps)

    # take means by bands
    bands_means = [_calc_bands_mean(fft_amp) for fft_amp in fft_amps]  # 1345

    # take means by subject
    FFT_bands_mean_sub = {}
    for sub in dlf.db.index:
        data_all["FFT_bands_mean"] = bands_means
        # dlf.db.index
        indi = data_all["subject"] == sub
        FFT_bands_mean_sub[sub] = np.array(data_all[indi]["FFT_bands_mean"]).mean()

    data_uq = (
        data_all[["subject", "disease_type"]].drop_duplicates().set_index("subject")
    )

    mm = np.array([FFT_bands_mean_sub[sub] for sub in list(data_uq.index)])
    dd = np.array(list(data_uq["disease_type"]))
    ss = np.array(list(data_uq.index))
    assert len(mm) == len(dd) == len(ss)
    return mm, dd, ss


def run_ANOVA_and_plot_topomaps(mm, dd, ss, montage, BANDS_LIM_HZ_DICT, sdir):
    ## Executes ANOVA
    dfs = []
    for i_ch in range(mm.shape[1]):
        for i_band in range(mm.shape[2]):
            df = pd.DataFrame()
            df["Mean FFT amp"] = mm[:, i_ch, i_band]
            df["Diagnosis"] = dd
            aov = pg.anova(
                dv="Mean FFT amp", between="Diagnosis", data=df, detailed=True
            )  # .round(3)

            aov["Montage"] = montage[i_ch]
            aov["Freq. Band"] = list(BANDS_LIM_HZ_DICT.keys())[i_band]
            dfs.append(aov)
    dfs = pd.concat(dfs).set_index(["Montage", "Freq. Band"])
    dfs_simple = dfs[["F", "p-unc", "np2"]].dropna()

    ## Plots F-values on topomaps
    F_vals, F_montage_wo_ref, F_bands_uq = extract_a_kind_of_value("F", dfs_simple)
    p_unc_vals, p_montage_wo_ref, p_bands_uq = extract_a_kind_of_value(
        "p-unc", dfs_simple
    )
    np2_vals, np2_montage_wo_ref, np2_bands_uq = extract_a_kind_of_value(
        "np2", dfs_simple
    )

    # Bonferroni correction
    n_tests = mm.shape[1] * mm.shape[2]
    p_corr_vals = p_unc_vals * n_tests

    mngs.io.save(dfs, sdir + "anova_all.csv")
    mngs.io.save(dfs_simple, sdir + "anova_simple.csv")

    fig, axes = utils.plot_topomap_bands(
        F_vals,
        F_montage_wo_ref,
        rename_band_names(F_bands_uq),
        unit_label="F-value",
    )
    # fig.show()
    mngs.io.save(fig, sdir + "F_val_on_topomaps.png")
    plt.close()

    ## eta squared (as an effect size indicator) on topomap
    fig, axes = utils.plot_topomap_bands(
        np2_vals,
        np2_montage_wo_ref,
        rename_band_names(np2_bands_uq),
        unit_label="eta squared",
        vmin=0,
        vmax=0.25,
    )
    mngs.io.save(fig, sdir + "np2_val_on_topomaps.tif")
    plt.close()

    ## p-val on topomap
    # alpha = 0.05
    # signi = (p_corr_vals < alpha).astype(int) # significant channels are complicated to drew
    fig, axes = utils.plot_topomap_bands(
        p_corr_vals,
        p_montage_wo_ref,
        rename_band_names(p_bands_uq),
        unit_label="p-corrected",
        vmin=0.0,
        vmax=1.0,
    )
    mngs.io.save(fig, sdir + "p_corr_val_on_topomaps.png")
    plt.close()

    ## each disease type
    for dt in ["HV", "AD", "DLB", "NPH"]:
        indi = dd == dt
        log10_mm_dt = np.log10(mm[indi].mean(axis=0) + 1e-5)
        fig, axes = utils.plot_topomap_bands(
            log10_mm_dt,
            p_montage_wo_ref,
            rename_band_names(p_bands_uq),
            unit_label="log10(FFT amp)",
            vmax=-0.66,
            vmin=-1.65,
        )
        fig.suptitle(dt)
        mngs.io.save(fig, sdir + f"log10_FFT_amp_{dt}_on_topomaps.png")
        plt.close()

    for dt in ["HV", "AD", "DLB", "NPH"]:
        indi = dd == dt
        mm_dt = mm[indi].mean(axis=0)
        fig, axes = utils.plot_topomap_bands(
            mm_dt,
            p_montage_wo_ref,
            rename_band_names(p_bands_uq),
            unit_label="FFT amp",
            vmax=0.22,
            vmin=0.025,
        )
        fig.suptitle(dt)
        mngs.io.save(fig, sdir + f"FFT_amp_{dt}_on_topomaps.png")
        plt.close()


if __name__ == "__main__":
    ## Fix seed
    mngs.general.fix_seeds(seed=42, np=np)

    # ## configure matplotlib
    # utils.general.configure_mpl(plt)

    ################################################################################
    ## Parameters
    ################################################################################
    disease_types = ["HV", "AD", "NPH", "DLB"]
    samp_rate = 500
    montage = mngs.io.load("./config/load_params.yaml")["montage"]
    BANDS_LIM_HZ_DICT = mngs.io.load("./config/global.yaml")["BANDS_LIM_HZ_DICT"]

    ## Calculates means of FFT amplitude by frequency bands
    mm_Osk, dd_Osk, ss_Osk = calc_bands_mean_of_a_dataset(
        "BIDS_dataset_Osk_v5.3", disease_types, samp_rate
    )
    mm_Kochi, dd_Kochi, ss_Kochi = calc_bands_mean_of_a_dataset(
        "BIDS_dataset_v1.1_Kochi", disease_types, samp_rate
    )
    mm_Nissei, dd_Nissei, ss_Nissei = calc_bands_mean_of_a_dataset(
        "BIDS_dataset_Nissei_v1.1", disease_types, samp_rate
    )

    ## Runs ANOVA
    run_ANOVA_and_plot_topomaps(
        mm_Osk,
        dd_Osk,
        ss_Osk,
        montage,
        BANDS_LIM_HZ_DICT,
        f"./results/FFT_topomap/Osaka/",
    )
    run_ANOVA_and_plot_topomaps(
        mm_Kochi,
        dd_Kochi,
        ss_Kochi,
        montage,
        BANDS_LIM_HZ_DICT,
        f"./results/FFT_topomap/Kochi/",
    )
    run_ANOVA_and_plot_topomaps(
        mm_Nissei,
        dd_Nissei,
        ss_Nissei,
        montage,
        BANDS_LIM_HZ_DICT,
        f"./results/FFT_topomap/Nissei/",
    )

    for i_ch in range(len(montage)):
        for i_band in range(len(BANDS_LIM_HZ_DICT)):
            mm_Osk_ch_band = mm_Osk[:, i_ch, i_band]
            mm_Kochi_ch_band = mm_Kochi[:, i_ch, i_band]
            mm_Nissei_ch_band = mm_Nissei[:, i_ch, i_band]

            fig, ax = plt.subplots()
            ch_name = mngs.general.connect_strs(montage[i_ch], filler="-")
            band_name = list(BANDS_LIM_HZ_DICT.keys())[i_band]
            institution = (
                ["Osaka" for _ in range(len(dd_Osk))]
                + ["Kochi" for _ in range(len(dd_Kochi))]
                + ["Nissei" for _ in range(len(dd_Nissei))]
            )
            df = pd.DataFrame(
                {
                    "mean_FFT_amp": np.hstack(
                        [mm_Osk_ch_band, mm_Kochi_ch_band, mm_Nissei_ch_band]
                    ),
                    "disease_type": np.hstack([dd_Osk, dd_Kochi, dd_Nissei]),
                    "institution": institution,
                }
            )
            ax = sns.boxplot(
                x="disease_type",
                y="mean_FFT_amp",
                hue="institution",
                data=df,
            )
            ax.set_ylim(0.01, 0.45)
            fig.suptitle(f"{ch_name}\n{band_name}")
            mngs.io.save(
                fig, f"./results/FFT_topomap/mean_FFT_amp/{i_band}_{band_name}_ch#{i_ch:02d}.png"
            )
            plt.close()

    ## EOF
