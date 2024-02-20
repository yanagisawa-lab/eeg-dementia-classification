#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-02-20 22:03:40 (ywatanabe)"

import torch
from eeg_dementia_classification_MNet import MNet_1000

# Parameters
DISEASE_TYPES = ["HV", "AD", "DLB", "NPH"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNet
model = MNet_1000(DISEASE_TYPES, is_ensemble=True).to(device)

# Generates demo signals
bs, n_chs, seq_len = 16, 19, 1000
x = torch.rand(bs, n_chs, seq_len).to(device)

# Feeds the demo signals
y = model(x)

# Print the output of the MNet_1000 model
print(y)


# EOF
