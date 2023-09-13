#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:45:43 (ywatanabe)"

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pingouin as pg
import seaborn as sns
from scipy.stats import ks_2samp
import pandas as pd
from glob import glob

# Loads
def print_med_iqr(df):
    med = df.describe()["50%"]
    iqr = float(np.abs(np.diff(df.describe()[["75%", "25%"]])))
    print(med, iqr)
                
def pearsons_chi2_test(df, x, y):
    _df = df.copy()
    _df["sex"] = _df["sex"].replace({-0.5: "Female", 0.5: "Male"})
    
    cross_table = pd.crosstab(_df[x], _df[y])
    
    expected, observed, stats = pg.chi2_independence(
        _df,
        x=x,
        y=y,
    )

    i_pearson = 0
    p = stats.pval[i_pearson]
    v = stats.cramer[i_pearson]
    return p, v, cross_table

def check(disease_types):
    plt.close()
    tgts = disease_types
    disease_types_str = mngs.general.connect_strs(disease_types, filler="_vs_")

    lpath = glob(
        f"./train/MNet_1000_subj/ywatanabe/submission_2023_0614/{disease_types_str}/"
        "_MNet*/subj-level/PassingClassifier_wo-reject-option/dMCI+Dementia/"
        "prediction_with_age_sex_MMSE.csv"
    )[0]

    ldir = mngs.general.split_fpath(lpath)[0]
    sdir = ldir + "correct_vs_incorrect/"
    
    df = mngs.io.load(lpath).set_index("subject")
    df["is_correct"] = df["is_correct"].astype(float)
    df["sex"] = df["sex"].replace({"F": -0.5, "M": 0.5})

    df["Prediction"] = df["is_correct"].replace(
        {
            0: "Incorrect",
            1: "Correct",
        }
    )
    df["Disease type"] = df["target"]
    df_all = pd.DataFrame()
    for var in ["age", "sex", "MMSE"]:
        indi_tgts = [df["Disease type"] == tgt for tgt in tgts]

        
        ps_ks, Ds, ps_chi, Vs, ns_correct, ns_incorrect = [], [], [], [], [], []
        for i_tgt, ii in enumerate(indi_tgts):

            # Pearson's Chi-squared test
            if var == "sex": # categorical
                p_chi, V, cross_table = pearsons_chi2_test(df[ii], var, "is_correct")
                mngs.io.save(cross_table, sdir + f"cross_table_{var}_{tgts[i_tgt]}.csv")
            else:
                p_chi, V = np.nan, np.nan
            ps_chi.append(p_chi)
            Vs.append(V)                
                
            
            # KS test
            samp_correct = df[ii][var][df[ii]["is_correct"] == 1]
            n_correct = len(samp_correct)
            
            samp_incorrect = df[ii][var][df[ii]["is_correct"] == 0]
            n_incorrect = len(samp_incorrect)

            ns_correct.append(n_correct)
            ns_incorrect.append(n_incorrect)            
            
            if n_correct * n_incorrect != 0:
            
                k2_2samp_out = ks_2samp(
                        samp_correct,
                        samp_incorrect,
                    )
            
                ps_ks.append(k2_2samp_out.pvalue)

                z = k2_2samp_out.statistic
                D = z / np.sqrt(n_correct * n_incorrect / (n_correct + n_incorrect))
                Ds.append(D)

            else:
                ps_ks.append(0)
                Ds.append(np.inf)


            # For details
            if disease_types_str == "HV_vs_AD+DLB+NPH":
                if i_tgt == 0: # HV
                    if var == "age":
                        print_med_iqr(samp_correct)
                        print_med_iqr(samp_incorrect)

                        # # HV_correct vs Dementia_correct
                        # samp_HV_correct = df[indi_tgts[0]][var][df[indi_tgts[0]]["is_correct"] == 1]
                        # samp_DEM_correct = df[indi_tgts[1]][var][df[indi_tgts[1]]["is_correct"] == 1]
                        # k2_2samp_out = ks_2samp(
                        #     samp_HV_correct,
                        #     samp_DEM_correct,
                        # )
                        # p = k2_2samp_out.pvalue
                        # z = k2_2samp_out.statistic
                        # D = z / np.sqrt(n_correct * n_incorrect / (n_correct + n_incorrect))
                        # print_med_iqr(samp_HV_correct)
                        # print_med_iqr(samp_DEM_correct)


            if disease_types_str == "AD_vs_DLB":
                if i_tgt == 1: # DLB
                    if var == "sex":
                        female_ratio_correct = 1 - (samp_correct + .5).describe()["mean"]
                        female_ratio_incorrect = 1 - (samp_incorrect + .5).describe()["mean"]
                        print(female_ratio_correct) #  0.439
                        print(female_ratio_incorrect) # 0.944

                        # data = pg.read_dataset("chi2_independence")
                        # data["sex"].value_counts(ascending=True)
                        # expected, observed, stats = pg.chi2_independence(data, x="sex", y="target")
                        
                        p, v, cross_table = pearsons_chi2_test(df[ii], var, "is_correct")
                        
                        samp_correct = df[ii][var][df[ii]["is_correct"] == 1]                        

            if disease_types_str == "HV_vs_AD_vs_DLB_vs_NPH":
                if i_tgt == 0: # HV
                    if var == "age":
                        import ipdb; ipdb.set_trace()                        
                        print_med_iqr(samp_correct)
                        print_med_iqr(samp_incorrect)


                
                
        ps_ks = np.array(ps_ks).round(3)
        Ds = np.array(Ds).round(3)
        ps_chi = np.array(ps_chi).round(3)
        Vs = np.array(Vs).round(3)        
        


            
            
        # plot
        fig, ax = plt.subplots()
        colors = [mngs.plt.colors.to_hex_code(c_str) for c_str in ["dan", "navy"]]
        sns.set_palette(sns.color_palette(colors))

        ax = sns.violinplot(
            x="Disease type",
            y=var,
            data=df,
            hue="Prediction",
            split=True,
            scale="count",
            order=tgts,
            hue_order=["Correct", "Incorrect"]
            # palette=colors,
        )
        handler, label = ax.get_legend_handles_labels()
        handler, label = [], []
        ax.legend(handler, label, loc="upper left", bbox_to_anchor=(1.05, 1))

        if var == "age":
            ylim = (40, 105)
        if var == "sex":
            ylim = (-1.5, 1.5)
        if var == "MMSE":
            ylim = (-5, 37)

        ax.set_ylim(ylim)
        fig.suptitle(var)
        mngs.io.save(fig, sdir + f"violinplot_{var}.png")

        # Table
        df_var = pd.DataFrame(
            {
                "Disease type": tgts,
                "var": var,                
                "ps_ks": ps_ks,
                "Ds": Ds,
                "ps_chi": ps_chi,
                "Vs": Vs,
                "n_incorrect": ns_incorrect,
                "n_correct": ns_correct,                
            }
        )
        df_all = pd.concat([df_all, df_var])

        # elligible: - 0.01
        # small: 0.01 - 0.20
        # medium: 0.20 - 0.50
        # Large: 0.80 -

        # spath_ks_test_csv = f"./tmp/Table_6/correct_vs_incorrect_ks_test_{var}.csv"
        # mngs.io.save(df_all, sdir + f"ks_test_{var}.csv")
        
    mngs.io.save(df_all, sdir + f"ks_test.csv")

    df_all["Classification task"] = disease_types_str
    df_all = df_all.set_index(["Classification task", "Disease type", "var"])
    
    return df_all, fig


if __name__ == "__main__":

    disease_types = [
        ["HV", "AD"],
        ["HV", "DLB"],
        ["HV", "NPH"],
        ["HV", "AD+DLB+NPH"],
        ["AD", "DLB"],
        ["AD", "NPH"],        
        ["DLB", "NPH"],
        ["AD", "DLB", "NPH"],
        ["HV", "AD", "DLB", "NPH"],
    ]

    df = pd.DataFrame()
    for dt in disease_types:
        df = pd.concat([df, check(dt)[0]], axis=0)

    mngs.io.save(df, "./results/tables/correct_incorrect_table.csv")
