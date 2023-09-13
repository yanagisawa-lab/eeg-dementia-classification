#!/usr/bin/env python

import os
from glob import glob

# import utils
import mngs
import pandas as pd
from natsort import natsorted
import numpy as np

# Functions
def get_ldir(
    comparison,
    model_name,
    task_name,
    window_size_sec,
    lroot_dir=None,
    wo_ws=False,
):

    ldir = os.path.join(
        lroot_dir,
        comparison,
        f"_{model_name}_WindowSize-{window_size_sec}-sec_*/subj-level/{task_name}/",
    )

    if wo_ws:
        ldir = ldir.split("_WindowSize")[0]

    return ldir


def get_metrics(ldir):
    metrics = {}
    try:
        ts = ldir.split("_")[-1].split("/")[0]
        metrics["Time Stamp"] = ts
    except:
        metrics["Time Stamp"] = None

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
    comparison,
    model_name,
    task_name,
    window_size_sec,
    ldir,
    wo_ws=False,
    subj_level_dict=None,
):

    print("\nldir:\n{}\n".format(ldir))

    metrics_dict = get_metrics(ldir)

    df.loc["Task Name", comparison] = task_name
    df.loc["Model Name", comparison] = model_name
    df.loc["Window Size [sec]", comparison] = window_size_sec

    for k, v in metrics_dict.items():
        df.loc[k, comparison] = v

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

    # metrics_dict = get_metrics(ldir)
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

    # for k, v in metrics_dict.items():
    #     df.loc[k, comparison] = v

    df_merged = pd.concat([df_merged, dfs], axis=1)
    return df_merged


if __name__ == "__main__":
    # Paraeters
    COMPARISONS_ALL = [
        "HV_vs_AD+DLB+NPH",
        "HV_vs_AD",
        "HV_vs_DLB",
        "HV_vs_NPH",
        "AD_vs_DLB",
        "AD_vs_NPH",
        "DLB_vs_NPH",
        "AD_vs_DLB_vs_NPH",
        "HV_vs_AD_vs_DLB_vs_NPH",
    ]

    # Summarize results
    _lroot_dir = "./2_train_and_eval/MNet_1000_subj/ywatanabe/submission_2022_0919/"
    model_name = "MNet_1000"
    tasks = [
        "dMCI+Dementia",
        "cMCI_single",
        "cMCI_ensemble",
        "Kochi_single",
        "Kochi_ensemble",
        "Nissei_single",
        "Nissei_ensemble",
        "Nissei_single_cMCI",
        "Nissei_ensemble_cMCI",
    ]

    window_size_sec = "2.0"
    dfs_metrics = []
    dfs_clf_rep = []
    for task_name in tasks:
        df_metrics = pd.DataFrame()
        df_clf_rep = pd.DataFrame()
        for comparison in COMPARISONS_ALL:

            if (task_name == "cMCI_ensemble") & ("HV" in comparison):
                continue
            if (task_name == "cMCI_single") & ("HV" in comparison):
                continue

            out = get_ldir(
                comparison,
                model_name,
                task_name,
                window_size_sec,
                lroot_dir=_lroot_dir,
                wo_ws=False,
            )

            for lroot_dir_tmp in natsorted(glob(out)):
                if "no_sig_False_no_age_True_no_sex_True_no_MMSE_True" in lroot_dir_tmp:
                    lroot_dir = lroot_dir_tmp
                    break

            # print(lroot_dir)
            df_metrics = pack_metrics_to_df(
                df_metrics,
                comparison,
                model_name,
                task_name,
                window_size_sec,
                lroot_dir,
                subj_level_dict=None,
            )

            df_clf_rep = pack_clf_rep_to_df(
                df_clf_rep,
                comparison,
                model_name,
                task_name,
                window_size_sec,
                lroot_dir,
                subj_level_dict=None,
            )

            try:
                if len(natsorted(glob(out))) == 0:
                    break

                for lroot_dir in natsorted(glob(out)):
                    if "no_sig_False_no_age_True_no_sex_True_no_MMSE_True" in lroot_dir:
                        break

                # print(lroot_dir)
                df_metrics = pack_metrics_to_df(
                    df_metrics,
                    comparison,
                    model_name,
                    task_name,
                    window_size_sec,
                    lroot_dir,
                    subj_level_dict=None,
                )

            except Exception as e:
                print(e)

            try:
                df_clf_rep = pack_clf_rep_to_df(
                    df_clf_rep,
                    comparison,
                    model_name,
                    task_name,
                    window_size_sec,
                    lroot_dir,
                    subj_level_dict=None,
                )
            except Exception as e:
                print(e)

        dfs_metrics.append(df_metrics)
        dfs_clf_rep.append(df_clf_rep)

    df_all = pd.concat(dfs_metrics, axis=1)
    df_clf_rep = pd.concat(dfs_clf_rep, axis=1)

    mngs.io.save(df_all.T, "./results/tables/scalar_metrics.csv")
