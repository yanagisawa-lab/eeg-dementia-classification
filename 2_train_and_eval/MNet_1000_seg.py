#!/usr/bin/env python

import inspect
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import re
import warnings
from glob import glob

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(".")
from eeg_dementia_classification import utils

# sys.path.append("./externals/")
from eeg_dementia_classification.externals import ranger
from sklearn.metrics import balanced_accuracy_score

sys.path.append("./externals")
from dataloader.DataLoaderFiller import DataLoaderFiller


pd.set_option("display.max_columns", 10)

try:
    from pandas.core.common import SettingWithCopyWarning
except:
    from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


################################################################################
## Functions
################################################################################
def determine_save_dir(
    disease_types,
    model_name,
    window_size_sec,
    max_epochs,
    no_sig,
    no_age,
    no_sex,
    no_MMSE,
    no_mtl,
):
    sdir = mngs.general.mk_spath("")
    comparison = mngs.general.connect_strs(disease_types, filler="_vs_")
    sdir = (
        sdir + f"{comparison}/"
        f"_{model_name}_WindowSize-{window_size_sec}-sec_MaxEpochs_{max_epochs}"
        f"_no_sig_{no_sig}"
        f"_no_age_{no_age}"
        f"_no_sex_{no_sex}"
        f"_no_MMSE_{no_MMSE}"
        f"_no_mtl_{no_mtl}"
        f"_{mngs.general.gen_timestamp()}/seg-level/"
    )
    return sdir


def load_model_and_model_config():
    from eeg_dementia_classification.models.MNet.MNet_1000 import MNet_1000 as Model

    FPATH_MODEL = inspect.getfile(Model)
    FPATH_MODEL_CONF = FPATH_MODEL.replace(".py", ".yaml")
    MODEL_CONF = mngs.general.load(FPATH_MODEL_CONF)
    return Model, MODEL_CONF, FPATH_MODEL, FPATH_MODEL_CONF


def load_global_config(FPATH_GLOBAL_CONF):
    GLOBAL_CONF = {"SAMP_RATE": mngs.general.load(FPATH_GLOBAL_CONF)["SAMP_RATE"]}
    return GLOBAL_CONF


def load_dataloader_config(FPATH_DL_CONF):
    DL_CONF = mngs.general.load(FPATH_DL_CONF)
    return DL_CONF


def define_parameters(is_debug_mode=False):
    ## Default config files

    # global
    FPATH_GLOBAL_CONF = "eeg_dementia_classification/config/global.yaml"
    GLOBAL_CONF = load_global_config(FPATH_GLOBAL_CONF)

    # model
    Model, MODEL_CONF, FPATH_MODEL, FPATH_MODEL_CONF = load_model_and_model_config()

    ## dataloader
    # load_params
    FPATH_LOAD_CONF = "eeg_dementia_classification/config/load_params.yaml"
    LOAD_CONF = mngs.io.load(FPATH_LOAD_CONF)

    # filler_params
    FPATH_FILLER_CONF = "eeg_dementia_classification/config/filler_params.yaml"
    FILLER_CONF = mngs.io.load(FPATH_FILLER_CONF)

    default_confs = {
        "default_global_conf.yaml": GLOBAL_CONF,
        "default_model_conf.yaml": MODEL_CONF,
        "default_load_conf.yaml": LOAD_CONF,
        "default_filler_conf.yaml": FILLER_CONF,
    }

    ## Merges all default configs
    merged_conf = utils.merge_dicts_without_overlaps(
        GLOBAL_CONF, MODEL_CONF, LOAD_CONF, FILLER_CONF
    )

    ## Verifies n_gpus
    merged_conf["n_gpus"] = utils.verify_n_gpus(merged_conf["n_gpus"])

    ## Updates merged_conf
    window_size_sec = float(
        np.round(merged_conf["window_size_pts"] / merged_conf["SAMP_RATE"], 1)
    )
    batch_size = (
        merged_conf["batch_size"] * merged_conf["n_gpus"]
        if not is_debug_mode
        else 16 * merged_conf["n_gpus"]
    )
    lr = float(merged_conf["lr"]) * merged_conf["n_gpus"]

    sdir = determine_save_dir(
        args.disease_types,
        Model.__name__,
        window_size_sec,
        args.max_epochs,
        args.no_sig,
        args.no_age,
        args.no_sex,
        args.no_MMSE,
        args.no_mtl,
    )

    # ss, ee = re.search("2022-[0-9]{4}-[0-9]{4}", sdir).span()
    # if len(glob(sdir.replace(sdir[ss:ee], "*"))) > 0:
    #     print("\nSkipped\n")
    #     exit()

    sdir_wo_time = sdir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    merged_conf["montage"] = [
        f"{bi[0]}-{bi[1]}" for bi in merged_conf["montage"]
    ]  # fixme
    assert len(
        mngs.general.search(args.montages_to_mask, merged_conf["montage"])[0]
    ) == len(args.montages_to_mask)
    merged_conf.update(
        {
            "MAX_EPOCHS": args.max_epochs,
            "device_type": device.type,
            "device_index": device.index,
            "sdir": sdir,
            "is_debug_mode": is_debug_mode,
            "window_size_sec": window_size_sec,
            "batch_size": batch_size,
            "lr": lr,
            "disease_types": args.disease_types,
            "dataset_ver": "5.3",
            "no_sig": args.no_sig,
            "no_age": args.no_age,
            "no_sex": args.no_sex,
            "no_MMSE": args.no_MMSE,
            "no_mtl": args.no_mtl,
            "montages_to_mask": args.montages_to_mask,
        }
    )

    ## Saves files to reproduce
    files_to_reproduce = [
        mngs.io.get_this_fpath(when_ipython="/dev/null"),
        FPATH_MODEL,
        FPATH_MODEL_CONF,
        FPATH_LOAD_CONF,
        FPATH_FILLER_CONF,
    ]

    for f in files_to_reproduce:
        mngs.io.save(f, merged_conf["sdir"])

    return merged_conf, default_confs, Model


def init_a_model(Model, config):
    model = Model(config)

    if config["n_gpus"] > 1:
        model = nn.DataParallel(model)
        print(f'Let\'s use {config["n_gpus"]} GPUs!')
    return model


def train_and_validate(
    dlf,
    i_fold,
    Model,
    merged_conf,
):
    print(f"\n {'-'*40} fold#{i_fold} starts. {'-'*40} \n")

    # model
    device = f"{merged_conf['device_type']}:{merged_conf['device_index']}"
    dlf.fill(i_fold, reset_fill_counter=True)  # to get n_subjs_tra
    merged_conf["n_subjs_tra"] = len(
        dlf.subs_trn
    )  # len(np.unique(dlf.dl_tra.dataset.arrs_list[-1]))
    model = init_a_model(Model, merged_conf).to(device)
    mtl = utils.MultiTaskLoss(are_regression=[False, False]).to(device)
    optimizer = ranger.Ranger(
        list(model.parameters()) + list(mtl.parameters()), lr=merged_conf["lr"]
    )

    # starts the current fold's loop
    i_global = 0
    lc_logger = mngs.ml.LearningCurveLogger()
    early_stopping = utils.EarlyStopping(patience=50, verbose=True)
    for i_epoch, epoch in enumerate(tqdm(range(merged_conf["MAX_EPOCHS"]))):

        dlf.fill(i_fold, reset_fill_counter=False)

        step_str = "Validation"
        for i_batch, batch in enumerate(dlf.dl_val):
            _, loss_diag_val = utils.base_step(
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
                no_mtl=args.no_mtl,
                print_batch_interval=False,
            )
        lc_logger.print(step_str)

        step_str = "Training"
        for i_batch, batch in enumerate(dlf.dl_tra):
            optimizer.zero_grad()
            loss, _ = utils.base_step(
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
                no_mtl=args.no_mtl,
                print_batch_interval=False,
            )
            loss.backward()
            optimizer.step()
            i_global += 1
        lc_logger.print(step_str)

        bACC_val = np.array(lc_logger.logged_dict["Validation"]["bACC_diag_plot"])[
            np.array(lc_logger.logged_dict["Validation"]["i_epoch"]) == i_epoch
        ].mean()

        model_spath = (
            merged_conf["sdir"]
            + f"checkpoints/model_fold#{i_fold}_epoch#{i_epoch:03d}.pth"
        )
        mtl_spath = model_spath.replace("model_fold", "mtl_fold")
        spaths_and_models_dict = {model_spath: model, mtl_spath: mtl}

        early_stopping(loss_diag_val, spaths_and_models_dict, i_epoch, i_global)
        # early_stopping(-bACC_val, spaths_and_models_dict, i_epoch, i_global)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    true_label_val = np.hstack(
        lc_logger.get_x_of_i_epoch(
            "true_label_diag", "Validation", early_stopping.i_epoch
        )
    )

    pred_proba_val = np.vstack(
        lc_logger.get_x_of_i_epoch(
            "pred_proba_diag", "Validation", early_stopping.i_epoch
        )
    )
    pred_class_val = pred_proba_val.argmax(axis=-1)

    bACC_val_best = balanced_accuracy_score(true_label_val, pred_class_val)

    to_test = [i_global, early_stopping, dlf, lc_logger, merged_conf]
    return bACC_val_best, to_test
    # return None, None


def test(
    i_fold, Model, reporter, i_global, early_stopping, dlf, lc_logger, merged_conf
):
    model = init_a_model(Model, merged_conf)
    model.load_state_dict(
        torch.load(list(early_stopping.spaths_and_models_dict.keys())[0])
    )
    mtl = utils.MultiTaskLoss(are_regression=[False, False])
    mtl.load_state_dict(
        torch.load(list(early_stopping.spaths_and_models_dict.keys())[1])
    )

    device = f"{merged_conf['device_type']}:{merged_conf['device_index']}"
    model.to(device)
    mtl.to(device)

    step_str = "Test"
    for i_batch, batch in enumerate(dlf.dl_tes):

        _, _ = utils.base_step(
            step_str,
            model,
            mtl,
            batch,
            device,
            i_fold,
            early_stopping.i_epoch,
            i_batch,
            early_stopping.i_global,
            lc_logger,
            no_mtl=args.no_mtl,
            print_batch_interval=False,
        )
    lc_logger.print(step_str)

    ## Evaluate on Test dataset
    true_class_tes = np.hstack(lc_logger.dfs["Test"]["true_label_diag"])
    pred_proba_tes = np.vstack(lc_logger.dfs["Test"]["pred_proba_diag"])
    pred_class_tes = pred_proba_tes.argmax(axis=-1)

    labels = list(
        dlf.cv_dict["label_int_2_conc_class_dict"].values()
    )  # dlf.disease_types
    reporter.calc_metrics(
        true_class_tes, pred_class_tes, pred_proba_tes, labels=labels, i_fold=i_fold
    )

    ## learning curves
    plt_config_dict = dict(
        dpi=300,
        figsize=(16.2, 10),
        # figscale=1.0,
        figscale=2.0,
        fontsize=16,
        labelsize="same",
        legendfontsize="xx-small",
        tick_size="auto",
        tick_width="auto",
        hide_spines=False,
    )
    lc_fig = lc_logger.plot_learning_curves(
        plt,
        title=(
            f"fold#{i_fold}\nmax epochs: {merged_conf['MAX_EPOCHS']}\n"
            f"window_size: {merged_conf['window_size_sec']} [sec]"
        ),
        plt_config_dict=plt_config_dict,
    )
    reporter.add("learning_curve", lc_fig)

    return reporter


def main(is_debug_mode=False):
    merged_conf, default_confs, Model = define_parameters(is_debug_mode=is_debug_mode)

    sys.stdout, sys.stderr = mngs.general.tee(sys, sdir=merged_conf["sdir"])
    print(f"no_mtl: {args.no_mtl}")
    mngs.general.fix_seeds(np=np, torch=torch)

    reporter = mngs.ml.ClassificationReporter(merged_conf["sdir"])

    dlf = DataLoaderFiller(
        "./data/BIDS_Osaka",
        args.disease_types,
        drop_cMCI=True,
    )

    # k-fold CV loop
    for i_fold in range(merged_conf["num_folds"]):

        bACC_val_best, to_test = train_and_validate(
            dlf,
            i_fold,
            Model,
            merged_conf,
        )
        reporter = test(i_fold, Model, reporter, *to_test)

    ## Saves the results
    reporter.summarize()
    reporter.save(meta_dict={**default_confs, "merged_conf.yaml": merged_conf})
    confmat_plt_config = dict(
        figsize=(15, 15),
        # labelsize=8,
        # fontsize=6,
        # legendfontsize=6,
        figscale=2,
        tick_size=0.8,
        tick_width=0.2,
    )

    sci_notation_kwargs = dict(
        order=3,  # 1
        fformat="%1.0d",
        scilimits=(-3, 3),
        x=False,
        y=True,
    )  # "%3.1f"

    reporter.plot_and_save_conf_mats(
        plt,
        extend_ratio=0.8,
        confmat_plt_config=confmat_plt_config,
        sci_notation_kwargs=sci_notation_kwargs,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="description")
    ap.add_argument(
        "-dts",
        "--disease_types",
        default=["HV", "AD", "DLB", "NPH"],
        nargs="*",
        help="HV AD DLB NPH",
    )

    ap.add_argument("-me", "--max_epochs", default=50, type=int, help=" ")  # 50

    ap.add_argument("--no_sig", default=False, action="store_true", help=" ")
    ap.add_argument("--no_age", default=False, action="store_true", help=" ")
    ap.add_argument("--no_sex", default=False, action="store_true", help=" ")
    ap.add_argument("--no_MMSE", default=False, action="store_true", help=" ")

    ap.add_argument(
        "--montages_to_mask",
        default=[],
        choices=[
            "FP1-A1",
            "F3-A1",
            "C3-A1",
            "P3-A1",
            "O1-A1",
            "FP2-A2",
            "F4-A2",
            "C4-A2",
            "P4-A2",
            "O2-A2",
            "F7-A1",
            "T7-A1",
            "P7-A1",
            "F8-A2",
            "T8-A2",
            "P8-A2",
            "Fz-A1",
            "Cz-A1",
            "Pz-A1",
        ],
        nargs="*",
        help="HV AD DLB NPH",
    )

    ap.add_argument("--no_mtl", default=False, action="store_true", help=" ")
    args = ap.parse_args()

    main(is_debug_mode=False)

    """
    screen
    spy train/MNet_1000_seg.py
    """
