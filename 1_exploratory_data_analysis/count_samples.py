#!/usr/bin/env python3
# Time-stamp: "2023-09-13 13:56:51 (ywatanabe)"

import sys

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
from ylab_dataloaders.eeg_dem_clf.v5.load import load_BIDS



## Loads
data_all_df = load_BIDS("./data/BIDS_dataset")


## Gets the first row of each subject
subj_uq = data_all_df["subject"].unique()
indi_first = [
    np.where((data_all_df["subject"] == su) is True)[0][0]
    for su in data_all_df["subject"].unique()
]
data_subj = data_all_df.iloc[indi_first]
del data_subj["eeg"]


## Counts disease type and disease level.
disease_types_uq = data_subj["disease_type"].unique()  # ['HV', 'AD', 'DLB', 'NPH']
disease_levels_uq = data_subj["disease_level"].unique()  # ["Dementia", "HV"]
pet_uq = data_subj["pet"].unique()  # [nan, "P", "N"]
Ns_disease_type_dict = {
    dt: (data_subj["disease_type"] == dt).sum() for dt in disease_types_uq
}
Ns_disease_level_dict = {
    dl: (data_subj["disease_level"] == dl).sum() for dl in disease_levels_uq
}
mngs.plt.configure_mpl(plt, figscale=1.5)

################################################################################
## disease type vs level
################################################################################
vs_disease_level_df = pd.DataFrame(index=disease_types_uq, columns=disease_levels_uq)
for dt in disease_types_uq:
    for dl in disease_levels_uq:
        vs_disease_level_df.loc[dt, dl] = (
            (data_subj["disease_type"] == dt) & (data_subj["disease_level"] == dl)
        ).sum()

## Plots
COLORS_DICT = {"Dementia": "red", "HV": "gray"}
fig_vs_level, ax_vs_level = plt.subplots()
for i_l, l in enumerate(vs_disease_level_df.columns):
    c = mngs.plt.colors.to_RGBA(COLORS_DICT[str(l)], alpha=0.6)
    if i_l == 0:
        accum = 0 * vs_disease_level_df[l].copy()
        ax_vs_level.bar(
            vs_disease_level_df.index,
            vs_disease_level_df[l],
            bottom=accum,
            label=l,
            color=c,
        )
    else:
        ax_vs_level.bar(
            vs_disease_level_df.index,
            vs_disease_level_df[l],
            bottom=accum,
            label=l,
            color=c,
        )
    accum += vs_disease_level_df[l].copy()
ax_vs_level.legend()
ax_vs_level.set_title("Disease level")
ax_vs_level.set_xlabel("Disease type")
ax_vs_level.set_ylabel("# subjects")
ax_vs_level.set_ylim(0, 110)
mngs.plt.ax_extend(ax_vs_level, 0.75, 0.75)
fig_vs_level.show()
mngs.io.save(fig_vs_level, "./results/figs/bar/fig_vs_level.png")


################################################################################
## disease type vs pet
################################################################################
vs_pet_df = pd.DataFrame(index=disease_types_uq, columns=pet_uq)
for dt in disease_types_uq:
    for p in pet_uq:
        p = str(p)
        if p == "nan":
            vs_pet_df.loc[dt, np.nan] = (
                ((data_subj["disease_type"] == dt) & data_subj["pet"].isna())
                .sum()
                .astype(int)
            )

        else:
            vs_pet_df.loc[dt, p] = (
                (data_subj["disease_type"] == dt) & (data_subj["pet"] == p)
            ).sum()

## Plots
COLORS_DICT = {"nan": "gray", "P": "red", "N": "blue"}
fig_vs_pet, ax_vs_pet = plt.subplots()
for i_l, l in enumerate(vs_pet_df.columns):
    c = mngs.plt.colors.to_RGBA(COLORS_DICT[str(l)], alpha=0.6)
    if i_l == 0:
        accum = 0 * vs_pet_df[l].copy()
        ax_vs_pet.bar(
            vs_pet_df.index,
            vs_pet_df[l],
            bottom=accum,
            label=l,
            color=c,
        )
    else:
        ax_vs_pet.bar(
            vs_pet_df.index,
            vs_pet_df[l],
            bottom=accum,
            label=l,
            color=c,
        )
    accum += vs_pet_df[l].copy()
ax_vs_pet.legend()
ax_vs_pet.set_title("PET label")
ax_vs_pet.set_xlabel("Disease type")
ax_vs_pet.set_ylabel("# subjects")
ax_vs_pet.set_ylim(0, 110)
mngs.plt.ax_extend(ax_vs_pet, 0.75, 0.75)
fig_vs_pet.show()
mngs.io.save(fig_vs_pet, "./results/figs/bar/fig_vs_pet.png")

## EOF
