#!/usr/bin/env python

import os
from glob import glob

# import utils
import mngs
import pandas as pd
from natsort import natsorted
import numpy as np


# Functions
def get_balanced_accuracy(lpath):
    balanced_accs = mngs.io.load(lpath)

    key_cv_mean = mngs.general.search(
        ["-folds_CV_mean"], list(balanced_accs["Unnamed: 0"])
    )[1][0]
    key_cv_std = mngs.general.search(
        ["-fold_CV_std"], list(balanced_accs["Unnamed: 0"])
    )[1][0]

    balanced_acc_mean = balanced_accs[balanced_accs["Unnamed: 0"] == key_cv_mean][
        "balanced_acc"
    ].iloc[0]

    balanced_acc_std = balanced_accs[balanced_accs["Unnamed: 0"] == key_cv_std][
        "balanced_acc"
    ].iloc[0]

    if balanced_acc_std == np.nan:
        balanced_acc_std = 0.0

    return float(balanced_acc_mean), float(balanced_acc_std)

def get_roc_auc(lpath, avg_method="macro"):
    aucs = mngs.io.load(lpath)

    i_mean = mngs.general.search(["-folds_CV_mean"], aucs["Unnamed: 0"])[0][0]
    i_std = mngs.general.search(["-fold_CV_std"], aucs["Unnamed: 0"])[0][0]    

    roc_auc_mean = aucs.iloc[i_mean]["roc/macro"]
    roc_auc_std = aucs.iloc[i_std]["roc/macro"]

    # try:
    #     roc_auc_std = aucs[i_std]["roc/macro"].iloc[0]
    # except Exception as e:
    #     print(e)
    #     roc_auc_std = np.nan
        
    return float(roc_auc_mean), float(roc_auc_std)

def get_metrics(ldir):
    metrics = {}

    try:
        metrics["Balanced Accuracy"] = "{:.3f} +/- {:.3f}".format(
            *get_balanced_accuracy(ldir + "balanced_acc.csv")
        )
    except:
        metrics["Balanced Accuracy"] = None

    try:
        metrics["ROC AUC (macro avg.)"] = "{:.3f} +/- {:.3f}".format(
            *get_roc_auc(ldir + "roc/macro.csv")
        )
    except:
        metrics["ROC AUC (macro avg.)"] = None

    try:
        metrics["PRE-REC AUC (macro avg.)"] = "{:.3f} +/- {:.3f}".format(
            *get_pr_auc(ldir + "pre_rec/macro.csv")
        )
    except:
        metrics["PRE-REC AUC (macro avg.)"] = None

    return metrics

def get_roc_auc(lpath, avg_method="macro"):
    aucs = mngs.io.load(lpath)

    i_mean = mngs.general.search(["-folds_CV_mean"], aucs["Unnamed: 0"])[0][0]
    i_std = mngs.general.search(["-fold_CV_std"], aucs["Unnamed: 0"])[0][0]    

    roc_auc_mean = aucs.iloc[i_mean]["roc/macro"]
    roc_auc_std = aucs.iloc[i_std]["roc/macro"]

    return float(roc_auc_mean), float(roc_auc_std)


def get_pr_auc(lpath, avg_method="macro"):
    aucs = mngs.io.load(lpath)

    key_cv_mean = mngs.general.search(["-folds_CV_mean"], list(aucs["Unnamed: 0"]))[1][
        0
    ]
    key_cv_std = mngs.general.search(["-fold_CV_std"], list(aucs["Unnamed: 0"]))[1][0]

    pr_auc_mean = aucs[aucs["Unnamed: 0"] == key_cv_mean]["pre_rec/macro"].iloc[0]
    pr_auc_std = aucs[aucs["Unnamed: 0"] == key_cv_std]["pre_rec/macro"].iloc[0]
    return float(pr_auc_mean), float(pr_auc_std)


def get_clf_report(lpath):
    clf_report = mngs.io.load(lpath)

    i_col_bacc = np.where("balanced accuracy" == np.array(clf_report.iloc[0].name))[0][
        0
    ]

    columns = []
    indi = ["precision", "recall", "f1-score"]
    data = []

    for i_col in range(1, i_col_bacc):
        pos_cls = clf_report.iloc[0].name[i_col]

        pre_mm = clf_report.iloc[1].name[i_col]
        pre_ss = clf_report.iloc[8].name[i_col]

        rec_mm = clf_report.iloc[2].name[i_col]
        rec_ss = clf_report.iloc[9].name[i_col]

        f1_mm = clf_report.iloc[3].name[i_col]
        f1_ss = clf_report.iloc[10].name[i_col]

        pre_str = f"{float(pre_mm):.03f} +/- {float(pre_ss):.03f}"
        rec_str = f"{float(rec_mm):.03f} +/- {float(rec_ss):.03f}"
        f1_str = f"{float(f1_mm):.03f} +/- {float(f1_ss):.03f}"

        columns.append(pos_cls)
        data.append(
            (
                pre_str,
                rec_str,
                f1_str,
            )
        )
    df = pd.DataFrame(columns=columns, data=np.array(data).T, index=indi)

    return df


def pack_metrics_to_df(
    df,
    ldir,
    index,
):

    print("\nldir:\n{}\n".format(ldir))

    metrics_dict = get_metrics(ldir)
    df_new = pd.DataFrame(metrics_dict, index=[index])

    df = pd.concat([df, df_new])

    return df


def pack_clf_rep_to_df(
    df_merged,
    comparison,
    model_name,
    task_name,
    window_size_sec,
    ldir,
    wo_ws=False,
    subj_level_dict=None,
):

    print(f"\nldir:\n{ldir}\n")

    df = pd.DataFrame()
    df.loc["Task Name", comparison] = task_name
    df.loc["Model Name", comparison] = model_name
    df.loc["Window Size [sec]", comparison] = window_size_sec

    clf_rep = get_clf_report(ldir + "clf_report.csv")    
    dfs = [df.copy() for _ in range(len(clf_rep.columns))]

    for i_col, col in enumerate(clf_rep.columns):
        clf_rep[col]
        dfs[i_col].loc["Positive Class", comparison] = col
        for k in clf_rep[col].index:
            dfs[i_col].loc[k, comparison] = clf_rep.loc[k, col]

    dfs = pd.concat(dfs, axis=1)

    df_merged = pd.concat([df_merged, dfs], axis=1)
    return df_merged 

if __name__ == "__main__":
    import re
    import mngs
    
    VAR_MASKING_DIR = "./2_train_and_eval/MNet_1000_subj/submission_2022_0919/HV_vs_AD_vs_DLB_vs_NPH/"
    var_masking_dirs = glob(VAR_MASKING_DIR + "*/subj-level/dMCI+Dementia/")

    df = pd.DataFrame()    
    for ldir in var_masking_dirs:
        _no_vars_str = re.search("no_sig_.*\/subj-level", ldir)    
        ss, ee = _no_vars_str.span()
        no_vars_str = ldir[ss:ee].split("/")[0]

        df = pack_metrics_to_df(df, ldir, no_vars_str)

    mngs.io.save(df, "./results/tables/var_masking_results.csv")
