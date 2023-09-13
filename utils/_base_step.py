#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-06-05 19:55:11 (ywatanabe)"

import re
import warnings

import mngs
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torchsummary import summary
import torch


def base_step(
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
    no_mtl=False,        
    print_batch_interval=100,
):

    if re.search("^[Tt]rain", step_str):
        model.train()
        mtl.train()
    else:
        model.eval()
        mtl.eval()

    batch = [v.to(device) for v in batch]
    Xb, Tb, Sgb, Slb, Ab, Gb, Mb = batch

    # ## torch_summary
    # if i_epoch == i_batch == 0:
    #     summary(model, Xb, Ab, Gb, Mb)

    logits_diag, logits_subj = model(Xb, Ab, Gb, Mb)
    

    ## Diagnosis classification, the main task
    pred_proba_diag = F.softmax(logits_diag, dim=-1)
    pred_class_diag = pred_proba_diag.argmax(dim=-1)
    xentropy_criterion_diag = nn.CrossEntropyLoss(reduction="none")
    loss_diag = xentropy_criterion_diag(logits_diag, Tb)

    ## Subject classification, the second task, which should not be solved.
    pred_proba_subj = F.softmax(logits_subj, dim=-1)
    pred_class_subj = pred_proba_subj.argmax(dim=-1)
    xentropy_criterion_subj = nn.CrossEntropyLoss(reduction="none")
    if re.search("[Tt]rain", step_str):
        try:
            loss_subj = xentropy_criterion_subj(logits_subj, Slb)
            # RuntimeError: Expected floating point type for target with class probabilities, got Long            
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
    else:
        loss_subj = torch.zeros_like(loss_diag)

    ## Multi Task Loss
    if not no_mtl:
        loss_diag_scaled, loss_subj_scaled = mtl([loss_diag, loss_subj])
        loss_tot = loss_diag_scaled.mean() + loss_subj_scaled.mean()
    else:
        loss_tot = loss_diag.mean()

    # n_classes = pred_class_diag.shape[-1]
    # bACC_chance = 1. / n_classes

    ## Logging
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            bACC_diag = balanced_accuracy_score(
                pred_class_diag.squeeze().cpu().numpy(), Tb.cpu().numpy().squeeze()
            )
        except Exception as e:
            print(e)
            bACC_diag = 0  # bACC_chance

        try:
            bACC_subj = balanced_accuracy_score(
                pred_class_subj.squeeze().cpu().numpy(), Slb.cpu().numpy().squeeze()
            )
        except Exception as e:
            print(e)
            bACC_subj = 0  # bACC_chance

    log_dict = {
        "loss_tot_plot": loss_tot.item(),
        "loss_diag_plot": loss_diag.mean().item(),
        "loss_subj_plot": loss_subj.mean().item(),
        "bACC_diag_plot": bACC_diag,
        "bACC_subj_plot": bACC_subj,
        "pred_proba_diag": pred_proba_diag.detach().cpu().numpy(),
        "pred_proba_subj": pred_proba_subj.detach().cpu().numpy(),
        "true_label_diag": Tb.detach().cpu().numpy(),
        "true_label_subj": Slb.detach().cpu().numpy(),
        "true_label_subj_global": Sgb.detach().cpu().numpy(),
        "i_fold": i_fold,
        "i_epoch": i_epoch,
        "i_global": i_global,
    }
    lc_logger(log_dict, step=step_str)

    ## Print
    if print_batch_interval:
        if i_batch % print_batch_interval == 0:
            print_txt = (
                f"\n{step_str}, batch#{i_batch}\n"
                f"loss_tot: {loss_tot:.3f}, loss_diag: {loss_diag.mean():.3f}, loss_subj: {loss_subj.mean():.3f}\n"
                f"bACC_diag: {bACC_diag.mean():.3f}, bACC_subj: {bACC_subj.mean():.3f}\n"
            )
            print(mngs.general.squeeze_spaces(print_txt))

    return loss_tot, loss_diag.mean().item()
