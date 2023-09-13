#!/usr/bin/env python

import argparse
import os
import sys


def warn(*args, **kwargs):
    pass


import warnings
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

pd.set_option("display.max_columns", 10)

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(".")
sys.path.append("./externals")
import re
from copy import deepcopy

import mngs
from eeg_dem_clf import utils
from eeg_dem_clf.models.MNet.MNet_1000 import MNet_1000 as Model

from dataloader.DataLoaderFiller import DataLoaderFiller

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

## Functions
def define_dirs(
    disease_types,
    clf_str,
    ROOT_DIR="./eeg_dem_clf/train/MNet_1000_seg/ywatanabe/",
):
    comparison_str = mngs.general.connect_strs(disease_types, filler="_vs_")

    # ldirs = glob(f"{ROOT_DIR}submission_2022_0919/{comparison_str}/*/seg-level/")
    ldirs = glob(f"{ROOT_DIR}submission_2023_0614/{comparison_str}/*/seg-level/")    
    sdirs = [
        (
            ldir.replace("_seg", "_subj").replace("seg-level", "subj-level")
        )
        for ldir in ldirs
    ]
    return ldirs, sdirs


def determine_montages_to_mask(i_exp):
    df = mngs.io.load("./results/montages_to_mask.csv", index_col=0)

    row = df.iloc[i_exp][:-1]
    i_true = row.name
    montages_to_mask = list(np.array(row.index)[list(row)])

    return montages_to_mask


def mk_dl_cMCI_from_data_not_used(dlf, labels):
    # finds subs_data_not_used_and_MCI
    data_cMCI = dlf.data_not_used[dlf.data_not_used["cognitive_level"] == "cMCI"]
    ii, _ = mngs.general.search(
        list(dlf.cv_dict["conc_class_str_2_label_int_dict"].keys()), data_cMCI["disease_type"]
    )
    data_cMCI = data_cMCI.iloc[ii]

    # # finds HV in test dataset
    # if "HV" in labels:
    #     subs_test = dlf.subs_tes
    #     ii_test, _ = mngs.general.search(subs_test, list(dlf.data_all["subject"]))
    #     data_test = dlf.data_all.iloc[ii_test]
    #     assert (set(data_test["subject"]) - set(subs_test)) == set()
    #     data_test_HV = data_test[data_test["disease_type"] == "HV"]
    #     data = pd.concat([data_cMCI, data_test_HV])
    # else:
    #     data = data_cMCI
    data = data_cMCI    

    _subs = list(data["subject"])
    _labels = [dlf.cv_dict["conc_class_str_2_label_int_dict"][l] for l in list(data["disease_type"])]
    _subj_str2id = {s: i for i, s in enumerate(_subs)}

    dl_data_not_used_MCI_and_HV = dlf.crop_without_overlap(
        data,
        _subs,
        _labels,
        dlf.filler_params["window_size_pts"],
        _subj_str2id,
        _subj_str2id,
    )

    return dl_data_not_used_MCI_and_HV

def format_montage(montage):
    try:
        montage = [
            f"{bi[0]}-{bi[1]}" for bi in montage
        ]
    except:
        pass
    return montage


"""
def mk_dl_mci(dlf, cMCI_or_dMCI, drop_cMCI, is_classified, labels):
    if not drop_cMCI:
        return _mk_dl_mci_in_test(dlf, cMCI_or_dMCI, is_classified, labels)
    if drop_cMCI:
        return _mk_dl_mci_data_not_used(dlf, cMCI_or_dMCI, is_classified, labels)


def _mk_dl_mci_data_not_used(dlf, cMCI_or_dMCI, is_classified, labels):
    # finds subs_data_not_used_and_MCI
    data_MCI = dlf.data_not_used[dlf.data_not_used["cognitive_level"] == "cMCI"]
    ii, _ = mngs.general.search(
        list(dlf.cv_dict["conc_class_str_2_label_int_dict"].keys()), data_MCI["disease_type"]
    )
    data_MCI = data_MCI.iloc[ii]  # ["disease_type"]

    # finds HV in test dataset
    if "HV" in labels:
        subs_test = dlf.subs_tes
        ii_test, _ = mngs.general.search(subs_test, list(dlf.data_all["subject"]))
        data_test = dlf.data_all.iloc[ii_test]
        assert (set(data_test["subject"]) - set(subs_test)) == set()
        data_test_HV = data_test[data_test["disease_type"] == "HV"]
        data = pd.concat([data_MCI, data_test_HV])
    else:
        data = data_MCI

    _subs = list(data["subject"])
    _labels = [dlf.cv_dict["conc_class_str_2_label_int_dict"][l] for l in list(data["disease_type"])]
    _subj_str2id = {s: i for i, s in enumerate(_subs)}

    dl_data_not_used_MCI_and_HV = dlf.crop_without_overlap(
        data,
        _subs,
        _labels,
        dlf.filler_params["window_size_pts"],
        _subj_str2id,
        _subj_str2id,
    )

    return dl_data_not_used_MCI_and_HV


def _mk_dl_mci_in_test(dlf, cMCI_or_dMCI, is_classified, labels):

    # find subs_test_and_MCI
    subs_test = dlf.subs_tes
    ii_test, _ = mngs.general.search(subs_test, list(dlf.data_all["subject"]))
    data_test = dlf.data_all.iloc[ii_test]
    assert (set(data_test["subject"]) - set(subs_test)) == set()

    data_test_MCI = data_test[data_test["cognitive_level"] == cMCI_or_dMCI]

    if "HV" in labels:
        data_test_HV = data_test[data_test["disease_type"] == "HV"]
        data = pd.concat([data_test_MCI, data_test_HV])
    else:
        data = data_test_MCI

    _subs = list(data["subject"])
    _labels = [dlf.cv_dict["conc_class_str_2_label_int_dict"][l] for l in list(data["disease_type"])]
    _subj_str2id = {s: i for i, s in enumerate(_subs)}

    dl_test_MCI_and_HV = dlf.crop_without_overlap(
        data,
        _subs,
        _labels,
        dlf.filler_params["window_size_pts"],
        _subj_str2id,
        _subj_str2id,
    )

    return dl_test_MCI_and_HV
"""

def define_reject_rules(
    dlf,
    model,
    mtl,
    device,
    i_fold,
    i_epoch,
    i_global,
    lc_logger,
    reporter,
    clf_str,
    labels,
    n_ftrs=1,
    use_reject_option=False,
):
    """
    Arguments:
        dlf:
            DataLoaderFiller

        model:
            MNet

        mtl:
            MultiTaskLoss

        device:
            something like "cuda:0"

        i_fold:
            the current fold's index

        i_epoch:
            the current epoch number

        i_global:
            the global number of training iteration

        lc_logger:
            used to collect and log outputs of batches during the fold

        reporter:
            used to summarize and saves all outputs of "num_folds"

        clf_str:
            default: "RidgeClassifier"

    Returns:
        conf_low_thres_optimal:
            determined lower cutoff threshold for the maximum value of
            classification confidence (posterior probability)
            of each segment-level predictions

        perplexity_high_thres_optimal:
            determined upper cutoff threshold for the perplexity of
            classification confidence (posterior probability)
            of each segment-level predictions

        lc_logger:
            updated lc-logger
    """

    step_str = "Validation"
    for i_batch, batch in enumerate(dlf.dl_val):
        _ = utils.base_step(
            step_str,
            model,
            mtl,
            batch,
            device,
            i_fold,
            i_epoch,
            i_batch,
            i_global,
            lc_logger,
            print_batch_interval=False,
        )

    if not use_reject_option:
        conf_low_thres_optimal = 0.0
        perplexity_high_thres_optimal = 4.0

    elif use_reject_option:

        # Extracts necessary info
        subjs_val = np.hstack(lc_logger.dfs[step_str]["true_label_subj"])
        true_class_val = np.hstack(lc_logger.dfs[step_str]["true_label_diag"])
        pred_proba_val = np.vstack(lc_logger.dfs[step_str]["pred_proba_diag"])
        pred_class_val = pred_proba_val.argmax(axis=-1)

        ## Determines an optimal rejct rule
        ## grid search optimal thresholds that maximizes subject-level balanced ACC
        ## on validation datasets and leads good coverage
        mngs.plt.configure_mpl(plt, fontsize=12)

        (
            conf_low_thres_optimal,
            perplexity_high_thres_optimal,
            fig_reject_rule,
        ) = utils.grid_search_the_reject_rule(
            clf_str,
            true_class_val,
            pred_proba_val,
            pred_class_val,
            subjs_val,
            labels,
            n_ftrs=n_ftrs,
            i_fold=i_fold,
            use_class_weight=True,
        )

        reporter.add(
            "figs_reject_rule",
            fig_reject_rule,
        )
        plt.close()

        reporter.add(
            "conf_low_thres_optimal",
            conf_low_thres_optimal,
        )

        reporter.add(
            "perplexity_high_thres_optimal",
            perplexity_high_thres_optimal,
        )

    print(conf_low_thres_optimal)
    print(perplexity_high_thres_optimal)
    return conf_low_thres_optimal, perplexity_high_thres_optimal, lc_logger


def train_clf(
    clf_str,
    lc_logger,
    conf_low_thres_optimal,
    perplexity_high_thres_optimal,
    n_ftrs=1,
):
    """
    Trains a classifier (e.g., RidgeClassifier) using the outputs of the 1st model
    (= segment-level model, or MNet specifically) as features. When turned on,
    a reject option is applied. The rule is determined by utilizing Validation Data.

    Arguments:
        clf_str:
            default: "RidgeClassifier"

        lc_logger:
            used to collect and log outputs of batches during the fold

        conf_low_thres_optimal:
            determined lower cutoff threshold for the maximum value of
            classification confidence (posterior probability)
            of each segment-level predictions

        perplexity_high_thres_optimal:
            determined upper cutoff threshold for the perplexity of
            classification confidence (posterior probability)
            of each segment-level predictions

        n_ftrs:
           The number of features to use from each subject for the classifier.
           (default: 1, meaning "mean")

    Returns:
        clf:
            A fitted classifier.
    """

    # Extracts necessary info
    step_str = "Validation"
    subjs_val = np.hstack(lc_logger.dfs[step_str]["true_label_subj"])
    true_class_val = np.hstack(lc_logger.dfs[step_str]["true_label_diag"])
    pred_proba_val = np.vstack(lc_logger.dfs[step_str]["pred_proba_diag"])
    pred_class_val = pred_proba_val.argmax(axis=-1)

    # Applies a reject option
    (
        true_class_sv_val,
        pred_proba_sv_val,
        pred_class_sv_val,
        subjs_sv_val,
        survival_rate_subj_val,
    ) = utils.apply_a_reject_option(
        true_class_val,
        pred_proba_val,
        pred_class_val,
        subjs_val,
        conf_low_thres=conf_low_thres_optimal,
        perplexity_high_thres=perplexity_high_thres_optimal,
    )  # sv: survived

    # Converts into subject-level features
    X_val, y_val, subjs_val = utils.to_subj_ftrs_and_labels(
        true_class_sv_val, pred_proba_sv_val, subjs_sv_val, n_ftrs=n_ftrs
    )
    X_val, y_val, subjs_val = shuffle(X_val, y_val, subjs_val)

    # Trains a classifier for the subject-level prediction task
    cw = compute_class_weight("balanced", classes=np.unique(y_val), y=y_val)
    cw = {k: cw[i_k] for i_k, k in enumerate(np.unique(y_val))}
    clf_server = utils.ClfServer(class_weight=cw)

    # Checks wheather the classifier here has self.predict_proba method
    scaler = StandardScaler() if clf_str != "PassingClassifier" else None
    try:
        clf = clf_server(clf_str, scaler=scaler)
        _ = clf.predict_proba(X_val)
    except Exception as e:
        clf = clf_server(clf_str, scaler=scaler)
        clf = CalibratedClassifierCV(clf)
    clf.str = clf_str

    # Runs data augumentation when samples are insufficient
    for i_attempt in range(3):
        try:
            clf.fit(X_val, y_val)
            break
        except ValueError:
            X_val = np.vstack([X_val, X_val])
            y_val = np.hstack([y_val, y_val])

    return clf


def evaluate(
    step_str,
    dl,
    labels,
    model,
    mtl,
    device,
    i_fold,
    i_epoch,
    i_global,
    lc_logger,
    reporter,
    conf_low_thres_optimal,
    perplexity_high_thres_optimal,
    clf,
    n_ftrs=1,
):
    """
    Evaluates scores on Test Data and saves the results on sdir using lc_logger and reporter.

    Arguments:
        step_str:
            Example:
                "Test", "MCI", "MCI23", "Kochi", "Nissei"

        dl:
            DataLoader

        model:
            MNet

        mtl:
            MultiTaskLoss

        device:
            something like "cuda:0"

        i_fold:
            the current fold's index

        i_epoch:
            the current epoch number

        i_global:
            the global number of training iteration

        lc_logger:
            used to collect and log outputs of batches during the fold

        reporter:
            used to summarize and saves all outputs of "num_folds"

        conf_low_thres_optimal:
            determined lower cutoff threshold for the maximum value of
            classification confidence (posterior probability)
            of each segment-level predictions

        perplexity_high_thres_optimal:
            determined upper cutoff threshold for the perplexity of
            classification confidence (posterior probability)
            of each segment-level predictions

        clf:
            Trained classifier to use for subject-level classification.

    Returns:
        None. However, reporter will save results to disks. Besides,
        MCI_dict will pass outputs as a reference. (Elements of list is mutable.)
    """

    for i_batch, batch in enumerate(dl):
        _ = utils.base_step(
            step_str,
            model,
            mtl,
            batch,
            device,
            i_fold,
            i_epoch,
            i_batch,
            i_global,
            lc_logger,
            print_batch_interval=False,
        )

    # Extracts necessary info
    subjs_step = np.hstack(lc_logger.dfs[step_str]["true_label_subj"])
    # subjs_global_step = np.hstack(lc_logger.dfs[step_str]["true_label_subj_global"])
    true_class_step = np.hstack(lc_logger.dfs[step_str]["true_label_diag"])
    pred_proba_step = np.vstack(lc_logger.dfs[step_str]["pred_proba_diag"])
    pred_class_step = pred_proba_step.argmax(axis=-1)

    # Applies a reject option; e.g., step is test
    (
        true_class_sv_step,
        pred_proba_sv_step,
        pred_class_sv_step,
        subjs_sv_step,
        survival_rate_subj_step,
    ) = utils.apply_a_reject_option(
        true_class_step,
        pred_proba_step,
        pred_class_step,
        subjs_step,
        conf_low_thres=conf_low_thres_optimal,
        perplexity_high_thres=perplexity_high_thres_optimal,
    )

    reporter.add(
        f"survival_rate_subj_{step_str}",
        survival_rate_subj_step,
    )
    print(f"\nsurvival_rate_subj_{step_str} (= coverage): {survival_rate_subj_step}\n")

    # Subject-level prediction
    X_step, y_step, S_step = utils.to_subj_ftrs_and_labels(
        true_class_sv_step, pred_proba_sv_step, subjs_sv_step, n_ftrs=n_ftrs
    )

    print(f"\nClassifier for fold#{i_fold} was:\n{clf.str}\n")

    ## If you want to use clf as the second model
    y_pred_class_step = clf.predict(X_step)
    y_pred_proba_step = clf.predict_proba(X_step)

    reporter.calc_metrics(
        y_step,
        y_pred_class_step,
        y_pred_proba_step,
        labels=labels,
        i_fold=i_fold,
        show=True,
    )

    if step_str == "dMCI+Dementia":
        return S_step, np.array(labels)[y_step], np.array(labels)[y_pred_class_step]
    else:
        return None, None, None


def evaluate_ensemble(
    step_str,
    dl,
    labels,
    models,
    mtls,
    device,
    reporter,
    conf_low_thres_optimal,
    perplexity_high_thres_optimal,
    clf,
    n_ftrs=1,
):
    """
    Evaluates scores on Test Data and saves the results on sdir using lc_logger and reporter.

    Arguments:
        step_str:
            Example:
                "Test", "MCI", "MCI23", "Kochi", "Nissei"

        dl:
            DataLoader

        model:
            MNet

        mtl:
            MultiTaskLoss

        device:
            something like "cuda:0"

        i_fold:
            the current fold's index

        i_epoch:
            the current epoch number

        i_global:
            the global number of training iteration

        lc_logger:
            used to collect and log outputs of batches during the fold

        reporter:
            used to summarize and saves all outputs of "num_folds"

        conf_low_thres_optimal:
            determined lower cutoff threshold for the maximum value of
            classification confidence (posterior probability)
            of each segment-level predictions

        perplexity_high_thres_optimal:
            determined upper cutoff threshold for the perplexity of
            classification confidence (posterior probability)
            of each segment-level predictions

        clf:
            Trained classifier to use for subject-level classification.

    Returns:
        None. However, reporter will save results to disks. Besides,
        MCI_dict will pass outputs as a reference. (Elements of list is mutable.)
    """
    y_step_all = []
    y_pred_proba_step_all = []
    lc_loggers = []
    i_fold, i_epoch, i_global = 0, 0, 0  # meaningless
    for i_model, (model, mtl) in enumerate(zip(models, mtls)):
        lc_logger = mngs.ml.LearningCurveLogger()
        _dl = deepcopy(dl)
        for i_batch, batch in enumerate(_dl):
            _ = utils.base_step(
                step_str,
                model,
                mtl,
                batch,
                device,
                i_fold,
                i_epoch,
                i_batch,
                i_global,
                lc_logger,
                print_batch_interval=False,
            )
        lc_loggers.append(lc_logger)

        # Extracts necessary info
        subjs_step = np.hstack(lc_logger.dfs[step_str]["true_label_subj"])
        true_class_step = np.hstack(lc_logger.dfs[step_str]["true_label_diag"])
        pred_proba_step = np.vstack(lc_logger.dfs[step_str]["pred_proba_diag"])
        pred_class_step = pred_proba_step.argmax(axis=-1)

        # Applies a reject option; e.g., step is test
        (
            true_class_sv_step,
            pred_proba_sv_step,
            pred_class_sv_step,
            subjs_sv_step,
            survival_rate_subj_step,
        ) = utils.apply_a_reject_option(
            true_class_step,
            pred_proba_step,
            pred_class_step,
            subjs_step,
            conf_low_thres=conf_low_thres_optimal,
            perplexity_high_thres=perplexity_high_thres_optimal,
        )

        reporter.add(
            f"survival_rate_subj_{step_str}",
            survival_rate_subj_step,
        )
        print(
            f"\nsurvival_rate_subj_{step_str} (= coverage): {survival_rate_subj_step}\n"
        )

        # Subject-level prediction
        X_step, y_step, S_step = utils.to_subj_ftrs_and_labels(
            true_class_sv_step, pred_proba_sv_step, subjs_sv_step, n_ftrs=n_ftrs
        )

        print(f"\nClassifier for fold#{i_fold} was:\n{clf.str}\n")

        ## If you want to use clf as the second model
        y_pred_class_step = clf.predict(X_step)
        y_pred_proba_step = clf.predict_proba(X_step)

        y_step_all.append(y_step)
        y_pred_proba_step_all.append(y_pred_proba_step)

    for i in range(len(y_step_all)):
        assert np.all(y_step_all[0] == y_step_all[1])

    y_pred_proba_step_mean = np.array(y_pred_proba_step_all).mean(axis=0)

    reporter.calc_metrics(
        y_step_all[0],
        y_pred_proba_step_mean.argmax(axis=-1),
        y_pred_proba_step_mean,
        labels=labels,
        i_fold=i_fold,
        show=True,
    )

def switch_tasks(disease_types, does_channel_exp, stop_bands):

    if does_channel_exp:
        return ["dMCI+Dementia"]

    if stop_bands is not None:
        return ["dMCI+Dementia"]

    else:
        # Switches tasks
        is_4_class_task = set(disease_types) == set(
            ["HV", "AD", "DLB", "NPH"]
        )
        # is_3_class_task = set(merged_conf["disease_types"]) == set(["AD", "DLB", "NPH"])
        is_HV_vs_Dementia_task = set(disease_types) == set(
            ["HV", "AD+DLB+NPH"]
        )

        # Main
        tgts =[]
        tgts += [
            "dMCI+Dementia",
        ]
        tgts += ["cMCI_single"]
        tgts += ["cMCI_ensemble"]                
        tgts += ["Kochi_single", "Kochi_ensemble"]
        tgts += ["Nissei_single", "Nissei_ensemble"]
        # tgts += ["Nissei_single_cMCI", "Nissei_ensemble_cMCI"]                

        # if not "HV" in set(disease_types):
        #     tgts += ["cMCI_single", "cMCI_ensemble"]


        

        # if is_4_class_task or is_HV_vs_Dementia_task:
        #     tgts += ["Kochi_single", "Kochi_ensemble"]
            # tgts += ["Nissei_single", "Nissei_ensemble"]
            # tgts += ["Kochi_ensemble"]
            # tgts += ["Nissei_ensemble"]

        # tgts += ["Kochi_ensemble"]            

        # tgts += ["Kochi_single"]        
        # tgts += ["Kochi_single", "Kochi_ensemble"]

        # tgts += ["Nissei_single", "Nissei_ensemble"]        
        
        return tgts
    

def main(
    IS_DEV_MODE=False,
    disease_types=["HV", "AD", "DLB", "NPH"],
    stop_bands=None,
):
    """
    1. Loads configurations when segment-level training was run

    2. Trains a scikit-learn model (RidgeClassifier) for
       the subject-level classification task

    3. Applies reject option while optimizing the rule using Validation Dataset

    4. Predicts and disease_types of Test Data

    5. Evaluates and saves the results

    6. Predict MCI subject with caring about data leakage

       Note: for unclassified data, "num_folds" patterns of models are avaraged
             and run as an ensemble model
    """

    # Preparation
    mngs.general.fix_seeds(seed=42, np=np, torch=torch)
    ldirs, sdirs = define_dirs(
        disease_types,
        clf_str,
        ROOT_DIR="./eeg_dem_clf/train/MNet_1000_seg/ywatanabe/",
    )

    for ldir, sdir in zip(ldirs, sdirs):

        # get info from ldir
        # variable-masking exp.
        is_sig_only = "no_sig_False_no_age_True_no_sex_True_no_MMSE_True" in ldir
        merged_conf = mngs.general.load(ldir + "merged_conf.yaml")
        # merged_conf["montage"] = format_montage(merged_conf["montage"])        

        # Frequency-band-masking exp.
        if stop_bands is not None:
            if not is_sig_only:
                print("\nSkipped due to not signal only trial.\n")            
                continue
            merged_conf.update({"stop_bands": stop_bands})
            stop_bands_str = mngs.general.connect_strs(stop_bands, filler="-")
            sdir += f"band_masking/{stop_bands_str}_stopped/"            

        # ch-masking exp.            
        if args.does_channel_masking_exp:
            if not is_sig_only:
                print("\nSkipped due to not signal only trial.\n")            
                continue
            sdir += "ch_masking/"
            n_exps = len(glob(sdir + "#*"))
            i_exp = n_exps + 1
            montages_to_mask = determine_montages_to_mask(i_exp)
            sdir += f"#{i_exp:07,}/"
        else:
            montages_to_mask = []
        merged_conf.update({"montages_to_mask": montages_to_mask})

        # # Skips if duplicated MNet
        # ss, ee = re.search("_MNet.*2022-[0-9]{4}-[0-9]{4}", sdir).span() # _MNet.*
        # is_duplicated = len(glob(sdir.replace(sdir[ss:ee], "*"))) > 0
        # if is_duplicated:
        #     print("\nSkipped\n")
        #     continue

        # Preparation
        sdir = "/results/" if IS_DEV_MODE else sdir
        sys.stdout, sys.stderr = mngs.general.tee(sys, sdir=sdir)        
        
        # Parameters
        device = f"{merged_conf['device_type']}:{merged_conf['device_index']}"
        n_rep = merged_conf["num_folds"] if not IS_DEV_MODE else 1
        tgts = switch_tasks(merged_conf["disease_types"], args.does_channel_masking_exp, stop_bands)
        mreporter = mngs.ml.MultiClassificationReporter(sdir, tgts=tgts)
        dlf = DataLoaderFiller(
            "./data/BIDS_Osaka",
            merged_conf["disease_types"],
            stop_bands=stop_bands,
            drop_cMCI=True,
        )

        models, mtls = [], []
        S_cv_all, T_cv_all, P_cv_all = [], [], []
        for i_fold in range(n_rep):
            print(f"\n {'-'*40} fold#{i_fold} starts. {'-'*40} \n")

            # preparation in this fold
            lc_logger = mngs.ml.LearningCurveLogger()
            dlf.fill(i_fold, reset_fill_counter=True) # Normal, cMCI, dMCI, and Dementia

            # Initializes models
            merged_conf["n_subjs_tra"] = len(dlf.subs_trn)
            model = Model(merged_conf)
            mtl = utils.MultiTaskLoss(are_regression=[False, False])
            weights_lpath = glob(ldir + f"checkpoints/model_fold#{i_fold}_epoch#*.pth")[
                0
            ]
            weights = mngs.general.load(weights_lpath)
            model.load_state_dict(weights)
            print(f"\nWeights have been loaded from {weights_lpath}\n")
            model.to(device)
            mtl.to(device)

            # for the ensemble model
            models.append(model)
            mtls.append(mtl)

            # Gets predictions on Validation Dataset
            labels = merged_conf["disease_types"]
            i_epoch = merged_conf["MAX_EPOCHS"] - 1  # not important
            i_global = 0  # not important

            # Applies reject option
            # (
            #     conf_low_thres_optimal,
            #     perplexity_high_thres_optimal,
            #     lc_logger,
            # ) = define_reject_rules(
            #     dlf,
            #     model,
            #     mtl,
            #     device,
            #     i_fold,
            #     i_epoch,
            #     i_global,
            #     lc_logger,
            #     mreporter.reporters[mreporter.tgt2id["dMCI+Dementia"]], # fixme
            #     clf_str,
            #     labels,
            #     n_ftrs=1,
            #     use_reject_option=use_reject_option,
            # )

            # Subject-level classifier
            clf = train_clf(
                clf_str,
                lc_logger,
                conf_low_thres_optimal,
                perplexity_high_thres_optimal,
                n_ftrs=1,
            )

            # Prediction on Test Dataset
            labels = list(dlf.cv_dict["label_int_2_conc_class_dict"].values())

            # Registers single-model tasks to run
            single_tasks_and_dls_list = []
            if "dMCI+Dementia" in tgts:
                single_tasks_and_dls_list.append(("dMCI+Dementia", dlf.dl_tes))

            if "cMCI_single" in tgts:
                try:
                    single_tasks_and_dls_list.append(("cMCI_single", dl_cMCI))
                except Exception as e:
                    dl_cMCI = mk_dl_cMCI_from_data_not_used(dlf, labels)
                    single_tasks_and_dls_list.append(("cMCI_single", dl_cMCI))

            if "Kochi_single" in tgts:
                try:
                    single_tasks_and_dls_list.append(("Kochi_single", dl_kochi))
                except Exception as e:
                    dl_kochi = utils.load_dl_kochi_or_nissei(
                        "BIDS_dataset_v1.1_Kochi",
                        disease_types,
                        from_pkl=True,
                    )  # fixme
                    single_tasks_and_dls_list.append(("Kochi_single", dl_kochi))

            if "Nissei_single" in tgts:
                try:
                    single_tasks_and_dls_list.append(("Nissei_single", dl_nissei))
                except Exception as e:
                    dl_nissei = utils.load_dl_kochi_or_nissei(
                        "BIDS_dataset_Nissei_v1.1",
                        disease_types,
                        from_pkl=True,
                    )
                    single_tasks_and_dls_list.append(("Nissei_single", dl_nissei))

            if "Nissei_single_cMCI" in tgts:
                try:
                    single_tasks_and_dls_list.append(("Nissei_single_cMCI", dl_nissei_MCI))
                except Exception as e:
                    dl_nissei_MCI = utils.load_dl_kochi_or_nissei(
                        "BIDS_dataset_Nissei_v1.1",
                        disease_types,
                        from_pkl=True,
                        Dementia_or_MCI="MCI"
                    )
                    single_tasks_and_dls_list.append(("Nissei_single_cMCI", dl_nissei_MCI))
                    
            # Runs single models
            if len(single_tasks_and_dls_list) != 0:
                for step_str, dl in single_tasks_and_dls_list:
                    _S, _T, _P = evaluate(
                        step_str,
                        dl,
                        labels,
                        model,
                        mtl,
                        device,
                        i_fold,
                        i_epoch,
                        i_global,
                        lc_logger,
                        mreporter.reporters[mreporter.tgt2id[step_str]],
                        conf_low_thres_optimal,
                        perplexity_high_thres_optimal,
                        clf,
                    )
                    if step_str == "dMCI+Dementia":
                        _S_str = np.array([dlf.sub_int2str_tes[s_str] for s_str in _S])
                        S_cv_all.append(_S_str)
                        T_cv_all.append(_T)
                        P_cv_all.append(_P)

        # For a later calculation: biases regarding age, sex, MMSE and predictions
        i_dMCI_Dementia = 0
        save_the_correct_or_incorrect_table_regarding_age_sex_and_MMSE(
            S_cv_all, T_cv_all, P_cv_all, sdir=mreporter.reporters[i_dMCI_Dementia].sdir
        )

        # Registers ensemble-model tasks to run
        ensemble_tasks_and_dls_list = []

        if "cMCI_ensemble" in tgts:
            try:
                ensemble_tasks_and_dls_list.append(("cMCI_ensemble", dl_cMCI))
            except Exception as e:
                dl_cMCI = mk_dl_cMCI_from_data_not_used(dlf, labels)
                ensemble_tasks_and_dls_list.append(("cMCI_ensemble", dl_cMCI))

        if "Kochi_ensemble" in tgts:
            try:
                ensemble_tasks_and_dls_list.append(("Kochi_ensemble", dl_kochi))
            except Exception as e:
                dl_kochi = utils.load_dl_kochi_or_nissei(
                    "BIDS_dataset_v1.1_Kochi",
                    disease_types,
                    from_pkl=True,
                )
                ensemble_tasks_and_dls_list.append(("Kochi_ensemble", dl_kochi))

        if "Nissei_ensemble" in tgts:
            try:
                ensemble_tasks_and_dls_list.append(("Nissei_ensemble", dl_nissei))
            except Exception as e:
                dl_nissei = utils.load_dl_kochi_or_nissei(
                    "BIDS_dataset_Nissei_v1.1",
                    disease_types,
                    from_pkl=True,
                )
                ensemble_tasks_and_dls_list.append(("Nissei_ensemble", dl_nissei))

        if "Nissei_ensemble_cMCI" in tgts:
            try:
                ensemble_tasks_and_dls_list.append(("Nissei_ensemble_cMCI", dl_nissei_MCI))
            except Exception as e:
                dl_nissei_MCI = utils.load_dl_kochi_or_nissei(
                    "BIDS_dataset_Nissei_v1.1",
                    disease_types,
                    from_pkl=True,
                    Dementia_or_MCI="MCI",
                )
                ensemble_tasks_and_dls_list.append(("Nissei_ensemble_cMCI", dl_nissei_MCI))
                
        # Runs ensemble models
        if len(ensemble_tasks_and_dls_list) != 0:
            for step_str, dl in ensemble_tasks_and_dls_list:
                evaluate_ensemble(
                    step_str,
                    dl,
                    labels,
                    models,
                    mtls,
                    device,
                    mreporter.reporters[mreporter.tgt2id[step_str]],
                    conf_low_thres_optimal,
                    perplexity_high_thres_optimal,
                    clf,
                    n_ftrs=1,
                )

        # Summarizes the results and saves them
        for tgt in tgts:
            mreporter.summarize(tgt=tgt)
            mreporter.save(meta_dict={}, tgt=tgt)

            confmat_plt_config = dict(
                figsize=(15, 15),
                # labelsize=8,
                # fontsize=6,
                # legendfontsize=6,
                figscale=2,
                tick_size=0.8,
                tick_width=0.2,
            )

            # sci_notation_kwargs = dict(
            #     order=3,  # 1
            #     fformat="%1.0d",
            #     scilimits=(-3, 3),
            #     x=False,
            #     y=True,
            # )  # "%3.1f"
            sci_notation_kwargs = None

            mreporter.plot_and_save_conf_mats(
                plt,
                extend_ratio=0.8,
                confmat_plt_config=confmat_plt_config,
                sci_notation_kwargs=sci_notation_kwargs,
                tgt=tgt,
            )


def save_the_correct_or_incorrect_table_regarding_age_sex_and_MMSE(
    S_cv_all,
    T_cv_all,
    P_cv_all,
    sdir="./results/",
):
    df = pd.DataFrame(
        {
            "subject": np.hstack(S_cv_all),
            "target": np.hstack(T_cv_all),
            "prediction": np.hstack(P_cv_all),
            "is_correct": np.hstack(T_cv_all) == np.hstack(P_cv_all),
        }
    )
    df = df.sort_values(["subject"]).reset_index()
    del df["index"]
    df = df.set_index("subject")

    df_stats = (
        mngs.io.load("./data/Dementia_EEGLIST_Handai-Anonym220805.xlsx")
        .set_index("EEG_ID")
        .sort_index()
    )

    for key in ["age", "sex", "MMSE"]:
        df[key] = df_stats.loc[df.index][key]

    mngs.io.save(df, f"{sdir}prediction_with_age_sex_MMSE.csv")


if __name__ == "__main__":
    from itertools import combinations
    
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-dts",
        "--disease_types",
        default=["HV", "AD+DLB+NPH"],
        # default=["HV", "AD", "DLB", "NPH"],
        # default=["AD", "DLB", "NPH"],
        # default=["HV", "NPH"],
        # default=["DLB", "NPH"],
        nargs="*",
        help=".",
    )

    ap.add_argument(
        "--does_channel_masking_exp",
        action="store_true",
        default=False,
        help=".",
    )

    ap.add_argument(
        "--does_freq_band_masking_exp",
        action="store_true",
        default=False,
        help=".",
    )
    
    ap.add_argument(
        "-sb",
        "--stop_bands",
        default=[], # fixme
        choices=["delta", "theta", "lalpha", "halpha", "beta", "gamma"],
        nargs="*",
        help=".",
    )
    
    args = ap.parse_args()

    ################################################################################
    ################################################################################
    disease_types = [
        # ["HV", "AD"],        
        # ["HV", "DLB"],
        # ["HV", "NPH"],
        # ["AD", "DLB"],
        ["AD", "NPH"],
        # ["DLB", "NPH"],
        # ["HV", "AD+DLB+NPH"],        
        # ["AD", "DLB", "NPH"],        
        # ["HV", "AD", "DLB", "NPH"],        
        ]

    for dt in disease_types:
        main(
            IS_DEV_MODE=False,
            disease_types=dt,
        )

    ################################################################################
    ################################################################################

    # # args.does_channel_masking_exp = True

    # # Frequency-band-masking exp
    # # args.does_freq_band_masking_exp = True # fixme
    # if args.does_freq_band_masking_exp:
    #     bands = ["delta", "theta", "lalpha", "halpha", "beta", "gamma"]
    #     all_stop_bands_comb = []
    #     for i in range(0, len(bands) + 1):
    #         all_stop_bands_comb += list(iter(combinations(bands, i)))
    #     all_stop_bands_comb[0] = None

    #     for stop_bands in all_stop_bands_comb[::-1]: # all_stop_bands_comb
    #         main(
    #             IS_DEV_MODE=False,
    #             disease_types=args.disease_types,
    #             clf_str=args.clf,
    #             stop_bands=stop_bands,
    #         )

    # else:
    #     main(
    #         IS_DEV_MODE=False,
    #         disease_types=args.disease_types,
    #         clf_str=args.clf,
    #         stop_bands=None,
    #     )

    main(
        IS_DEV_MODE=False,
        disease_types=args.disease_types,
        clf_str=args.clf,
        stop_bands=None,
    )
    
    """
    cudav=1
    spy ./train/MNet_1000_subj.py --does_channel_masking_exp
    """
