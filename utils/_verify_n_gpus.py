#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-06-23 20:03:38 (ywatanabe)"

import torch
import warnings

def verify_n_gpus(n_gpus):
    if torch.cuda.device_count() < n_gpus:
        warnings.warn(
            f"N_GPUS (= {n_gpus}) is larger "
            f"than n_gpus torch can access (= {torch.cuda.device_count()}). "
            f"Please check $CUDA_VISIBLE_DEVICES.",
            UserWarning,
        )
        return torch.cuda.device_count()
    
    else:
        return n_gpus
