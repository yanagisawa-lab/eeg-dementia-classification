#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 14:33:28 (ywatanabe)"

from glob import glob

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
import xgboost

df = mngs.io.load("./results/bands_stopping_exp.csv", index_col="index")
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

fig, ax = plt.subplots()
ax = sns.boxplot(
    data=abs_shap_df,
)

ax.set_title("Feature Importance")
mngs.io.save(fig, "./results/feature_importance_bands/fig.png")
mngs.io.save(abs_shap_df, "./results/feature_importance_bands/abs_shap_bands.csv")
plt.show()
