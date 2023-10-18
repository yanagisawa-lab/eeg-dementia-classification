#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-12 14:34:43 (ywatanabe)"

from eeg_dementia_classification_MNet import MNet_1000, EnsembleMNet_1000
import torch

## Single model
disease_types = ["HV", "AD", "DLB", "NPH"]
model = MNet_1000(disease_types)
model.load_weights(i_fold=0)

## Ensemble model
model = EnsembleMNet_1000(disease_types)
model.load_weights()

## Feeds data
bs, n_chs, seq_len = 16, 19, 1000
x = torch.rand(bs, n_chs, seq_len)
y = model(x)

## EOF

