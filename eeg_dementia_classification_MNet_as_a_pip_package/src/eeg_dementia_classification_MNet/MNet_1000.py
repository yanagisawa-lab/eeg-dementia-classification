#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-12 14:53:19 (ywatanabe)"

import torch
import torch.nn as nn
from eeg_dementia_classification_MNet._SingleMNet_1000 import SingleMNet_1000
from eeg_dementia_classification_MNet._EnsembleMNet_1000 import EnsembleMNet_1000

class MNet_1000(nn.Module):
    def __init__(self, disease_types, is_ensemble=False):
        super().__init__()
        self.is_ensemble = is_ensemble
        
        if is_ensemble:
            self.model = EnsembleMNet_1000(disease_types)
        else:
            self.model = SingleMNet_1000(disease_types)
            
    def load_weights(self, i_fold=0):
        if self.is_ensemble:
            self.model.load_weights()
        else:
            self.model.load_weights(i_fold=i_fold)
        return self.model

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Parameters
    DISEASE_TYPES = ["HV", "AD", "DLB", "NPH"]
    
    # Model
    model = MNet_1000(DISEASE_TYPES, is_ensemble=True)

    # Demo data
    bs, n_chs, seq_len = 16, 19, 1000
    x = torch.rand(bs, n_chs, seq_len)

    # Load pretrained weights
    model.load_weights()
    
    # Forward path
    y_diag = model(x)
