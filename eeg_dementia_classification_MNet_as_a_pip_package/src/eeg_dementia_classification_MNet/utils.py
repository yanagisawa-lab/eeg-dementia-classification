#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-11 16:01:04 (ywatanabe)"

def connect_strs(strs, filler="_"):  # connect_nums also works as connect_strs
    """
    Example:
        print(connect_strs(['a', 'b', 'c'], filler='_'))
        # 'a_b_c'
    """
    if isinstance(strs, list) or isinstance(strs, tuple):
        connected = ""
        for s in strs:
            connected += filler + s
        return connected[len(filler) :]
    else:
        return strs
