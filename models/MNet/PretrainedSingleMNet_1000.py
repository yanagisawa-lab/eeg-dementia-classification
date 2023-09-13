#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-07 12:56:30 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MNet.MNet_1000 import MNet_1000
from glob import glob
import sys

sys.path.append("./externals")
from dataloader.DataLoaderFiller import DataLoaderFiller
import mngs


class PretrainedSingleMNet_1000(nn.Module):
    """Loads a trained model in a fold out of five."""
    def __init__(
            self, disease_types, i_fold=0
    ):
        super().__init__()
        
        self.ldir = self.get_ldir(disease_types)
        
        self.merged_conf = mngs.general.load(self.ldir + "merged_conf.yaml")
        self.merged_conf["disease_types"] = disease_types

        self.model = self.load_a_model(self.ldir, i_fold)
        
    def get_ldir(self, disease_types):
        comparison_str = mngs.gen.connect_strs(disease_types, filler="_vs_")#"HV_vs_AD"
        ROOT_DIR = "./eeg_dementia_classification/train/MNet_1000_seg/ywatanabe/"
        ldir = glob(f"{ROOT_DIR}submission_2022_0919/{comparison_str}/*/seg-level/")[0]
        return ldir

    def load_a_model(self, ldir, i_fold):
        model = MNet_1000(self.merged_conf)
        weights = mngs.general.load(
            glob(ldir + f"checkpoints/model_fold#{i_fold}_epoch#*.pth")[0]
        )
        # Remove keys related to fc_subj
        weights = {k: v for k, v in weights.items() if not k.startswith('fc_subj')}        
        model.load_state_dict(weights, strict=False)
        return model

    def forward(self, x, Ab, Sb, Mb):
        y_diag, _ = self.model(x, Ab, Sb, Mb)
        pred_proba_diag = F.softmax(y_diag, dim=-1)
        return pred_proba_diag


if __name__ == "__main__":
    import torch

    # Parameters
    DISEASE_TYPES = ["HV", "AD", "DLB", "NPH"]
    
    # Model
    model = PretrainedSingleMNet_1000(DISEASE_TYPES)

    # Demo data
    bs, n_chs, seq_len = 16, 19, 1000
    inp = torch.rand(bs, n_chs, seq_len)
    Ab = torch.rand(
        bs,
    )
    Sb = torch.rand(
        bs,
    )
    Mb = torch.rand(
        bs,
    )

    # Forwarding data
    y_diag = model(inp, Ab, Sb, Mb)
