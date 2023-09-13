#!/usr/bin/env python3
# Time-stamp: "2021-11-30 11:11:01 (ylab)"

from time import sleep
from pprint import pprint


def get_params(model, tgt_name=None, sleep_sec=2, show=False):

    name_shape_dict = {}
    for name, param in model.named_parameters():
        if (tgt_name is not None) & (name == tgt_name):
            return param
        if tgt_name is None:
            # print(f"\n{param}\n{param.shape}\nname: {name}\n")
            if show is True:
                print(f"\n{param}: {param.shape}\nname: {name}\n")
                sleep(sleep_sec)
            name_shape_dict[name] = list(param.shape)

    if tgt_name is None:
        print()
        pprint(name_shape_dict)
        print()
