#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-07 12:56:07 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MNet.MNet_1000 import MNet_1000
from glob import glob
import sys

sys.path.append("./externals")
from dataloader.DataLoaderFiller import DataLoaderFiller
import mngs



class PretrainedEnsembleMNet_1000(nn.Module):
    def __init__(
            self, disease_types, save_activations=False,
    ):
        super().__init__()
        
        self.ldir = self.get_ldir(disease_types)
        
        self.merged_conf = mngs.general.load(self.ldir + "merged_conf.yaml")
        self.merged_conf["disease_types"] = disease_types
        self.merged_conf["save_activations"] = save_activations        

        self.models = [self.load_a_model(self.ldir, i_fold) for i_fold in range(5)]
        
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
        ys_diag = torch.stack([m(x, Ab, Sb, Mb)[0] for m in self.models], dim=0)
        preds_proba_diag = F.softmax(ys_diag, dim=-1)
        pred_proba_diag = preds_proba_diag.mean(axis=0)
        return pred_proba_diag


if __name__ == "__main__":
    import torch

    # Parameters
    DISEASE_TYPES = ["HV", "AD", "DLB", "NPH"]
    
    # Model
    model = PretrainedEnsembleMNet_1000(DISEASE_TYPES)

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

    # Forward path
    y_diag = model(inp, Ab, Sb, Mb)

