#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2022-07-03 12:38:11 (ywatanabe)"

import numpy as np
import mngs

def sort_disease_types(diags_list):
    diags_list = diags_list.copy()
    order = ["HV", "AD", "DLB", "NPH", "MCIsusp", "MCI", "PET-N", "PET-P"]

    _reordered_order = []
    for d in diags_list:
        try:
            i, m = mngs.general.grep(order, d)
            assert len(i) == 1
            _reordered_order.append(i[0])
        except Exception as e:
            print(e)

    reordered_order = np.argsort(_reordered_order)
    return [diags_list[i] for i in reordered_order]
