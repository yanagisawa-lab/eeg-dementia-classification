#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-09-13 13:51:06 (ywatanabe)"

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
from utils import load_data_all


## Functions
def _to_counted_table(data_uq):
    disease_types = list(data_uq["disease_type"].unique())
    cognitive_levels = list(data_uq["cognitive_level"].unique())
    df = pd.DataFrame(
        columns=disease_types,
        index=cognitive_levels,
    )
    for dt in disease_types:
        for cl in cognitive_levels:
            nn = (
                (data_uq["disease_type"] == dt) & (data_uq["cognitive_level"] == cl)
            ).sum()
            df.loc[cl, dt] = nn
    return df


def _sort_by_disease_type_and_cognitive_level(ser):
    df = pd.DataFrame(ser).reset_index()
    disease_types_order = ["HV", "AD", "DLB", "NPH"]
    cognitive_levels_order = ["Normal", "cMCI", "dMCI", "Dementia"]

    df_mapping = pd.DataFrame({"disease_types": disease_types_order})
    sort_mapping = df_mapping.reset_index().set_index("disease_types")
    df["disease_type_order"] = df["disease_type"].map(sort_mapping["index"])

    df_mapping = pd.DataFrame({"cognitive_level": cognitive_levels_order})
    sort_mapping = df_mapping.reset_index().set_index("cognitive_level")
    df["cognitive_level_order"] = df["cognitive_level"].map(sort_mapping["index"])

    df = df.sort_values(["disease_type_order", "cognitive_level_order"]).reset_index()

    keys_to_del = ["disease_type_order", "cognitive_level_order", "index", "level_0"]
    for k in keys_to_del:
        try:
            del df[k]
        except:
            pass
    return df


def main(BIDS_DATASET_NAME, plot=False):
    ## Loads
    load_params, data_all, data_not_used = load_data_all(
        BIDS_DATASET_NAME,
        apply_notch_filter=True,
        from_pkl=True,
    )  # 484

    ## Drops unnecessary data
    data_uq = data_all[
        ["subject", "disease_type", "cognitive_level", "age", "sex", "MMSE"]
    ].drop_duplicates()  # , "cognitive_level",
    ii, _ = mngs.general.search(
        ["Normal", "MCI$", "MCI23", "Dementia"], data_uq["cognitive_level"]
    )
    data_uq = data_uq.iloc[ii]

    data_uq["sex"] += 0.5  # female: 1, male: 0

    counted_table = _to_counted_table(data_uq)
    print(counted_table)

    # rename MCI, MCI23 to cMCI and dMCI
    data_uq = data_uq.replace({"MCI": "cMCI", "MCI23": "dMCI"})
    data_uq = _sort_by_disease_type_and_cognitive_level(data_uq)

    # for sample size
    data_uq["n"] = 1

    disease_types_order = ["HV", "AD", "DLB", "NPH"]
    cognitive_levels_order = ["Normal", "cMCI", "dMCI", "Dementia"]

    if plot:
        ax = sns.boxplot(
            x="disease_type",
            y="age",
            hue="cognitive_level",
            data=data_uq,
            order=disease_types_order,
            hue_order=cognitive_levels_order,
        )
        plt.show()

    ns = data_uq.groupby(["disease_type", "cognitive_level"])["n"].sum()

    age_medians = (
        data_uq.groupby(["disease_type", "cognitive_level"])["age"]
        .median()
        .rename("Age Median")
        .round(1)
    )
    _age_Q3s = data_uq.groupby(["disease_type", "cognitive_level"])["age"].quantile(
        0.75
    )
    _age_Q1s = data_uq.groupby(["disease_type", "cognitive_level"])["age"].quantile(
        0.25
    )
    age_IQRs = (_age_Q3s - _age_Q1s).rename("Age IQR").round()

    female_ratio = (
        data_uq.groupby(["disease_type", "cognitive_level"])["sex"]
        .mean()
        .rename("female_ratio")
    )
    female_ratio = abs(1 - female_ratio).round(2)

    MMSE_medians = (
        data_uq.groupby(["disease_type", "cognitive_level"])["MMSE"]
        .median()
        .rename("MMSE Median")
        .round(1)
    )
    _MMSE_Q3s = data_uq.groupby(["disease_type", "cognitive_level"])["MMSE"].quantile(
        0.75
    )
    _MMSE_Q1s = data_uq.groupby(["disease_type", "cognitive_level"])["MMSE"].quantile(
        0.25
    )
    MMSE_IQRs = (_MMSE_Q3s - _MMSE_Q1s).rename("MMSE IQR").round()

    df = _sort_by_disease_type_and_cognitive_level(
        pd.concat(
            [ns, age_medians, age_IQRs, female_ratio, MMSE_medians, MMSE_IQRs], axis=1
        )
    )

    mngs.io.save(df, "./results/demographic_info/data.csv")

    return data_uq, df


if __name__ == "__main__":
    data_uq_osaka, df_osaka = main(
        mngs.io.load("./config/global.yaml")["BIDS_Osaka_NAME"]
    )
    data_uq_kochi, df_kochi = main(
        mngs.io.load("./config/global.yaml")["BIDS_Kochi_NAME"]
    )
    data_uq_nissei, df_nissei = main(
        mngs.io.load("./config/global.yaml")["BIDS_Nissei_NAME"]
    )

    mngs.io.save(df_osaka, "./results/demographic_info/Osaka/n_sex.csv")
    mngs.io.save(df_kochi, "./results/demographic_info/Kochi/n_sex.csv")
    mngs.io.save(df_nissei, "./results/demographic_info/Nissei/n_sex.csv")

    data_uqs = [data_uq_osaka, data_uq_kochi, data_uq_nissei]
    hospitals = ["Osaka", "Kochi", "Nissei"]

    for data_uq, hospital in zip(data_uqs, hospitals):
        for var in ["age", "MMSE"]:
            for dt in ["HV", "AD", "DLB", "NPH"]:
                df = data_uq[data_uq["disease_type"] == dt][
                    ["cognitive_level", var]
                ].pivot(columns=["cognitive_level"])

                mngs.io.save(
                    df,
                    f"./results/demographic_info/{table_str}/{hospital}/{var}/{dt}.csv",
                )

    ## EOF
