#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 13:41:03 (ywatanabe)"

import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("./externals")
from dataloader.DataLoaderFiller import DataLoaderFiller


## Functions
def transform_var(var_str):
    if var_str == "age":
        transform = dlf.transform_age
    if var_str == "sex":
        transform = dlf.transform_sex
    if var_str == "MMSE":
        transform = dlf.transform_MMSE
    var_trn = dlf.db.loc[dlf.subs_trn][var_str]
    var_normal_trn = transform(var_trn)
    var_val = dlf.db.loc[dlf.subs_val][var_str]
    var_normal_val = transform(var_val)

    if var_str == "MMSE":
        var_normal_trn = pd.DataFrame(data=var_normal_trn, index=var_trn.index).squeeze() 
        var_normal_val = pd.DataFrame(data=var_normal_val, index=var_val.index).squeeze() 
    return var_trn, var_normal_trn, var_val, var_normal_val


def plot_hist(var_str, orig_xlim, transformed_lim):

    var_trn, var_normal_trn, var_val, var_normal_val = transform_var(var_str)

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].hist(var_trn, label="training", color="blue")  # 156
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].set_xlim(orig_xlim)
    axes[0, 1].hist(var_normal_trn, label="training", color="orange")
    axes[0, 1].set_xlim(transformed_lim)
    axes[0, 1].legend(loc="upper right")
    axes[1, 0].hist(var_val, label="validation", color="blue")  # 52
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].set_xlim(orig_xlim)
    axes[1, 1].hist(var_normal_val, label="validation", color="orange")
    axes[1, 1].set_xlim(transformed_lim)
    axes[1, 1].legend(loc="upper right")
    fig.suptitle(var_str.capitalize())
    fig.supylabel("Count")
    axes[1, 0].set_xlabel(f"Original {var_str}")
    axes[1, 1].set_xlabel(f"Transformed {var_str}")
    # plt.show()
    return fig

def save_as_csv(var_str):
    var_trn, var_normal_trn, var_val, var_normal_val = transform_var(var_str)
    df = pd.DataFrame({
        "Training_Original": var_trn,
        "Training_Transformed": var_normal_trn,
        "Validation_Original": var_val,
        "Validation_Transformed": var_normal_val,
        })
    mngs.io.save(df, f"./results/figs/hist/transformation/{var_str}.csv")

if __name__ == "__main__":
    import mngs

    # loads
    dlf = DataLoaderFiller(
        "./data/BIDS_Kochi",
        ["HV", "AD", "DLB", "NPH"],
        drop_cMCI=True,
    )
    dlf.fill(i_fold=0, reset_fill_counter=True)

    # csv
    save_as_csv("age")
    save_as_csv("sex")
    save_as_csv("MMSE")    

    # Plots
    fig_age = plot_hist("age", (45, 105), (-5, 5))
    fig_sex = plot_hist("sex", (-1, 1), (-1, 1))
    fig_MMSE = plot_hist("MMSE", (0, 30), (-3, 3))

    mngs.io.save(fig_age, "./results/figs/hist/transformation/age.png")
    mngs.io.save(fig_sex, "./results/figs/hist/transformation/sex.png")
    mngs.io.save(fig_MMSE, "./results/figs/hist/transformation/MMSE.png")    

    for p in patches:
        print(p.get_xy())
        print(p.get_width())
        print(p.get_height())

    ## EOF
