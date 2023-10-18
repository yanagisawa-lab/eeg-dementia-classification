#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-10-18 13:35:09 (ywatanabe)"

import torch
import torch.nn as nn
import torch.nn.functional as F
from ._SingleMNet_1000 import SingleMNet_1000
# from eeg_dementia_classification_MNet._SingleMNet_1000 import SingleMNet_1000
from glob import glob
import sys


class EnsembleMNet_1000(nn.Module):
    def __init__(
            self, disease_types, save_activations=False,
    ):
        super().__init__()

        self.disease_types = disease_types

        # self.models = [SingleMNet_1000(disease_types) for _ in range(5)]
        self.models = nn.ModuleList([SingleMNet_1000(disease_types) for _ in range(5)])

    def load_weights(self,):
        for i_fold, model in enumerate(self.models):
            model.load_weights(i_fold)        
        # self.models = [model.load_weights(i_fold) for i_fold, model in enumerate(self.models)]
        return self.models

    def forward(self, x):
        ys_diag = torch.stack([m(x) for m in self.models], dim=0)
        pred_proba_diag = F.softmax(ys_diag, dim=-1).mean(axis=0)
        return pred_proba_diag


if __name__ == "__main__":
    # Parameters
    DISEASE_TYPES = ["HV", "AD", "DLB", "NPH"]
    
    # Model
    model = EnsembleMNet_1000(DISEASE_TYPES).cuda()

    # Demo data
    bs, n_chs, seq_len = 16, 19, 1000
    x = torch.rand(bs, n_chs, seq_len).cuda()

    # Load pretrained weights
    model.load_weights()
    
    # Forward path
    y_diag = model(x)


