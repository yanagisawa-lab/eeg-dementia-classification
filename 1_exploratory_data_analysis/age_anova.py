#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 12:20:12 (ywatanabe)"

import matplotlib

# from utils import load_data_all
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import re
from itertools import combinations

import joypy
import matplotlib.pyplot as plt
import mngs
import pingouin as pg


def load():
    # Loads
    df = mngs.io.load("./data/secret/Dementia_EEGLIST_Handai-Anonym220805.xlsx")

    # Drops unnecessary rows
    tgts_disease_types = ["HV", "AD", "DLB", "NPH"]
    tgts_cognitive_level = ["Normal", "Dementia", "MCI", "MCI23"]
    df = df.iloc[mngs.general.search(tgts_disease_types, df["Disease type"])[0]]
    df = df.iloc[mngs.general.search(tgts_cognitive_level, df["Cognitive level"])[0]]
    return df


def drop_youngstars(df):
    # Drops youngstars
    drop_condi = (df["age"] < 40) & (df["Disease type"] == "HV")
    df = df[~drop_condi]
    return df


def count_sex(df):
    # Adds counts on column names
    _df = df[["Cognitive level", "Disease type", "age", "sex"]]
    # Counts gender
    _df["n"] = 1
    _df_gender_count = pd.pivot_table(
        _df, index=["Disease type", "sex"], aggfunc="count"
    )["n"]

    _disease_types, _counts = np.unique(_df["Disease type"], return_counts=True)
    for i_dt, dt in enumerate(_disease_types):
        indi = mngs.general.search(_disease_types[i_dt], list(_df["Disease type"]))[0]
        FF, MM = _df_gender_count[dt]["F"], _df_gender_count[dt]["M"]
        _df["Disease type"].iloc[indi] = (
            _disease_types[i_dt] + f"\n(M={MM}; F={FF})"  # N={MM+FF};\n
        )
    del _df["n"]
    return _df


def plot_box(_df, pattern=None):
    def _inner_mk_order_list(_df):
        # mk order list
        order_keys = []
        for key in ["HV", "AD", "DLB", "NPH"]:
            try:
                order_keys.append(
                    _df["Disease type"].iloc[
                        mngs.general.search(f"^{key}", _df["Disease type"])[0][0]
                    ]
                )
            except Exception as e:
                print(e)

        return order_keys

    order_keys = _inner_mk_order_list(_df)

    fig, ax = plt.subplots()
    ax = sns.boxplot(
        x="Disease type",
        y="age",
        data=_df,
        order=order_keys,
    )
    fig.suptitle(pattern)
    # fig.show()
    mngs.io.save(fig, f"./results/age_anova/{pattern}/age_boxplot.png")
    plt.close()


def plot_joy(_df, pattern=None):
    fig, axes = joypy.joyplot(_df, by="Disease type", column="age")
    axes[-1].set_xlabel("Age")
    fig.suptitle(pattern)
    mngs.io.save(fig, f"./results/age_anova/{pattern}/age_joyplot.png")
    plt.close()


def kruskal_wallis_test(_df, pattern=None):
    # ANOVA
    kruskal_out = pg.kruskal(
        data=_df,
        dv="age",
        between="Disease type",
    )
    p = f"{kruskal_out['p-unc'][0]:.1e}"
    pattern += f" (p={p})"
    mngs.io.save(
        kruskal_out, f"./results/age_anova/{pattern}/age_kruskal_wallis.csv"
    )
    return p


df_orig = load()

# All
df = df_orig.copy()
_df = count_sex(df)
p = kruskal_wallis_test(_df, pattern="All")
plot_box(_df, pattern=f"All (p={p})")
plot_joy(_df, pattern=f"All (p={p})")


# HV wo under 40
df = df_orig.copy()
# drop
drop_condi = (df["age"] < 40) & (df["Disease type"] == "HV")
print(drop_condi.sum())
df = df[~drop_condi]
#
_df = count_sex(df)
p = kruskal_wallis_test(_df, pattern="wo_HV_under_40")
plot_box(_df, pattern=f"wo_HV_under_40 (p={p})")
plot_joy(_df, pattern=f"wo_HV_under_40 (p={p})")


################################################################################
# HV wo under 40 & Dementia over 80
df = df_orig.copy()
low_HV = 65
low_DEM = 65
high_DEM = 75
# drop
drop_condi = (df["age"] < low_HV) & (df["Disease type"] == "HV")
print(drop_condi.sum())
df = df[~drop_condi]

drop_condi = ((df["age"] < low_DEM) + (high_DEM < df["age"])) & (
    (df["Disease type"] == "AD")
    + (df["Disease type"] == "DLB")
    + (df["Disease type"] == "NPH")
)

print(drop_condi.sum())
df = df[~drop_condi]

_df = count_sex(df)

p = kruskal_wallis_test(
    _df, pattern=f"wo_HV_under_{low_HV}_nor_Dementia_under_{low_DEM}_over_{high_DEM}"
)

plot_box(
    _df,
    pattern=f"wo_HV_under_{low_HV}_nor_Dementia_under_{low_DEM}_over_{high_DEM} (p={p})",
)
plot_joy(
    _df,
    pattern=f"wo_HV_under_{low_HV}_nor_Dementia_under_{low_DEM}_over_{high_DEM} (p={p})",
)
################################################################################
# Dementia; MCI; MCI23
for Dementia_or_MCI in ["Dementia", "MCI", "MCI23"]:
    df = df_orig.copy()
    drop_condi = (df["Cognitive level"] != Dementia_or_MCI) * (
        df["Disease type"] != "HV"
    )
    df = df[~drop_condi]

    _df = count_sex(df)
    p = kruskal_wallis_test(_df, pattern=f"HV_and_{Dementia_or_MCI}")

    plot_box(_df, pattern=f"HV_and_{Dementia_or_MCI} (p={p})")
    plot_joy(_df, pattern=f"HV_and_{Dementia_or_MCI} (p={p})")
################################################################################
# Dementia; MCI; MCI23 wo HV
for Dementia_or_MCI in ["Dementia", "MCI", "MCI23"]:
    df = df_orig.copy()
    drop_condi = (
        df["Cognitive level"] != Dementia_or_MCI
    )  # * (df["Disease type"] != "HV")
    df = df[~drop_condi]

    _df = count_sex(df)
    p = kruskal_wallis_test(_df, pattern=f"HV_and_{Dementia_or_MCI}_wo_HV")

    plot_box(_df, pattern=f"HV_and_{Dementia_or_MCI}_wo_HV (p={p})")
    plot_joy(_df, pattern=f"HV_and_{Dementia_or_MCI}_wo_HV (p={p})")

################################################################################
# Dementia vs MCI
df = df_orig.copy()
drop_condi_DEM = (df["Cognitive level"] != "Dementia") * (df["Disease type"] != "HV")
drop_condi_MCI = (df["Cognitive level"] != "MCI") * (df["Disease type"] != "HV")
drop_condi_MCI23 = (df["Cognitive level"] != "MCI23") * (df["Disease type"] != "HV")
df_dem = df[~drop_condi_DEM]
df_mci = df[~drop_condi_MCI]
df_mci23 = df[~drop_condi_MCI23]

df_dict = dict(
    Dementia=df_dem,
    MCI=df_mci,
    MCI23=df_mci23,
)

for df1_str, df2_str in combinations(["Dementia", "MCI", "MCI23"], 2):
    print(f"\n{'-'*40}\n")
    print(f"{df1_str} vs {df2_str}")
    df1 = df_dict[df1_str]
    df2 = df_dict[df2_str]
    for dt in ["HV", "AD", "DLB", "NPH"]:
        print(dt)
        mwu_out = pg.mwu(
            df1[df1["Disease type"] == dt]["age"],
            df2[df2["Disease type"] == dt]["age"],
        )
        print(mwu_out)
    print(f"\n{'-'*40}\n")
