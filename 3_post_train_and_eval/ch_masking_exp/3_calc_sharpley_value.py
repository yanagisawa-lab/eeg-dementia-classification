#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:44:25 (ywatanabe)"

from glob import glob

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import shap

## Functions
import sklearn

# from sklearn.model_selection import train_test_split
import xgboost
import seaborn as sns
import utils

df = mngs.io.load("./results/channels_stopping_exp.csv", index_col="rand_index")
del df["Unnamed: 0"]

# Calculates sharpley value
X = df.iloc[:, :-1].astype(float).copy()
y = df.iloc[:, -1]

model = xgboost.XGBRegressor().fit(X, y)
explainer = shap.Explainer(model)

shap_values = explainer(X)
abs_shap = abs(shap_values.values)
abs_shap_df = pd.DataFrame(
    data=abs_shap,
    columns=list(X.columns),
)
columns = [
    "FP1-A1",
    "FP2-A2",
    "F7-A1",
    "F3-A1",
    "Fz-A1",
    "F4-A2",
    "F8-A2",
    "T7-A1",
    "C3-A1",
    "Cz-A1",
    "C4-A2",
    "T8-A2",
    "P7-A1",
    "P3-A1",
    "Pz-A1",
    "P4-A2",
    "P8-A2",
    "O1-A1",
    "O2-A2",
]
abs_shap_df = abs_shap_df[columns]


# boxplot
fig, ax = plt.subplots()
ax = sns.boxplot(
    data=abs_shap_df,
    # data=abs_shap_norm_df,
)
ax.set_ylabel("Feature Importance")
plt.xticks(rotation=90)
mngs.io.save(fig, "./results/figs/box/feature_importance_channels.png")
mngs.io.save(abs_shap_df, "./results/figs/box/abs_shap_channels.csv")
plt.show()

# topomap
med_abs_shap_df = abs_shap_df.median(axis=0)
CH_MONTAGES_WO_REF = [
    ch_montage.split("-")[0].capitalize() for ch_montage in list(med_abs_shap_df.index)
]  # CH_MONTAGES]
fontsize = 12
mngs.plt.configure_mpl(plt, fontsize=fontsize)
fig = utils.plot_topomap(
    med_abs_shap_df,
    CH_MONTAGES_WO_REF,
    vmin=med_abs_shap_df.min(),
    vmax=med_abs_shap_df.max(),
    cmap="coolwarm",
)

mngs.io.save(fig, f"./results/figs/heatmap/feature_importance_topomap_channels_fs_{fontsize}.png")
