#!/usr/bin/env python3

import numpy as np
import torch


def torch_to_arr(x):
    is_arr = isinstance(x, (np.ndarray, np.generic))
    if is_arr:  # when x is np.array
        return x
    if torch.is_tensor(x):  # when x is torch.tensor
        return x.detach().cpu().numpy()


def cvt_multi2single_model_state_dict(state_dict_multi):
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict_multi.items():
        name = k
        if name.startswith("module."):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict
    
