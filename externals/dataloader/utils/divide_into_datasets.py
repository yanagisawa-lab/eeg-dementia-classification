#!/usr/bin/env python3

import os
import random
import sys

import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.model_selection import StratifiedKFold, train_test_split

from .label_handlers import (_expand_conc_classes_str,
                             _mk_dict_for_conc_class_str_2_label_int,
                             _mk_dict_for_label_int_2_conc_class)


def divide_into_datasets(data_all, div_params):
    # val_ratio_to_trnのrange確認
    assert div_params["val_ratio_to_trn"] >= 0

    # シードを初期化
    random.seed(div_params["random_state"])
    np.random.seed(div_params["random_state"])

    # 変換の辞書を作成
    conc_class_str_2_label_int_dict = _mk_dict_for_conc_class_str_2_label_int(
        div_params["conc_classes"], delimiter="+"
    )
    label_int_2_conc_class_dict = _mk_dict_for_label_int_2_conc_class(
        div_params["conc_classes"]
    )

    # 被験者IDを取得
    all_subs = data_all["subject"]  # 334
    # クラス名を数値ラベルに変換して取得
    all_labels = [
        conc_class_str_2_label_int_dict.get(x, np.nan)
        for x in data_all[div_params["class_field_name"]]
    ]

    # 数値ラベルに変換できなかったデータを削除＆被験者IDとラベル名のペアをuniqueなものにする
    sub_info_pair = []
    for sub, label in zip(all_subs, all_labels):
        if np.isnan(label):
            continue
        sub_info_pair.append((sub, label))
    # 重複するペアを削除
    sub_info_pair = sorted(list(set(sub_info_pair)))

    # 被験者IDと数値ラベルの取得
    all_subs_uq = [x[0] for x in sub_info_pair]
    all_labels_uq = np.array([x[1] for x in sub_info_pair])

    # 被験者IDはこの時点でuniqueになっているはず
    assert len(set(all_subs_uq)) == len(all_subs_uq)

    if div_params["num_folds"] >= 2:
        skf = StratifiedKFold(
            n_splits=div_params["num_folds"],
            shuffle=True,
            random_state=div_params["random_state"],
        )
        # train+validationとtestに分割する
        div_trn_and_tes = [
            x for x in skf.split(np.zeros(all_labels_uq.shape[0]), all_labels_uq)
        ]

    if div_params["num_folds"] == 1:
        div_trn_and_tes = [(np.array(range(len(all_labels_uq))), [])]

    inds_tes = [x[1] for x in div_trn_and_tes]  # len: 3 -> 1
    if div_params["val_ratio_to_trn"] != 0:
        inds_trn = [None for _ in range(div_params["num_folds"])]
        inds_val = [None for _ in range(div_params["num_folds"])]
        for i, div_tmp in enumerate(div_trn_and_tes):
            inds_trn[i], inds_val[i] = train_test_split(
                div_tmp[0],
                test_size=div_params["val_ratio_to_trn"],
                stratify=all_labels_uq[div_tmp[0]],
                random_state=div_params["random_state"],
            )
    else:
        inds_trn = [x[0] for x in div_trn_and_tes]
        inds_val = [[] for _ in range(div_params["num_folds"])]

    if div_params["undersample_trn"]:
        print("\nPerforming undersampling on training data\n")
        inds_trn = [
            _under_sample_on_subj(x, all_labels_uq, label_int_2_conc_class_dict)
            for x in inds_trn
        ]
    if div_params["undersample_val"] and (div_params["val_ratio_to_trn"] != 0):
        print("\nPerforming undersampling on validation data\n")
        inds_val = [
            _under_sample_on_subj(x, all_labels_uq, label_int_2_conc_class_dict)
            for x in inds_val
        ]

    # 被験者IDを取得する．分かりやすさのために自然順ソートしておく
    subs_tes = [natsorted([all_subs_uq[i] for i in x]) for x in inds_tes]
    subs_trn = [natsorted([all_subs_uq[i] for i in x]) for x in inds_trn]
    subs_val = [
        natsorted([all_subs_uq[i] for i in x]) if len(x) != 0 else [] for x in inds_val
    ]

    for i in range(div_params["num_folds"]):
        assert len(set(subs_tes[i]) & set(subs_trn[i])) == 0
        assert len(set(subs_trn[i]) & set(subs_val[i])) == 0
        assert len(set(subs_val[i]) & set(subs_tes[i])) == 0

    for i in range(div_params["num_folds"]):
        for j in range(i + 1, div_params["num_folds"]):
            assert len(set(subs_tes[i]) & set(subs_tes[j])) == 0

    div_dict = {}
    div_dict["subs_trn"] = subs_trn
    div_dict["subs_val"] = subs_val
    div_dict["subs_tes"] = subs_tes
    div_dict["conc_class_str_2_label_int_dict"] = conc_class_str_2_label_int_dict
    div_dict["label_int_2_conc_class_dict"] = label_int_2_conc_class_dict

    return div_dict


def _under_sample_on_subj(inds, labels, label_int_2_conc_class_dict):
    target_labels = labels[inds]
    N_min = min(np.unique(target_labels, return_counts=True)[1])

    inds_new = []
    for x in label_int_2_conc_class_dict.keys():
        tmp_inds = inds[target_labels == x]
        np.random.shuffle(tmp_inds)
        inds_new.append(tmp_inds[0:N_min])

    return np.sort(np.concatenate(inds_new, axis=0))


if __name__ == "__main__":
    import sys

    sys.path.append("./bids/src")
    import mngs
    from ylab_dataloaders_fukuma.loader.loader_BIDS_simple import \
        loader as loader_func

    def load_BIDS(
        BIDS_ROOT,
        disease_types=["HV", "AD", "NPH", "DLB"],
        from_pkl=False,
        samp_rate=500,
    ):
        load_params = load_yaml("./config/load_params.yaml")
        load_params["BIDS_root"] = BIDS_ROOT
        load_params["conc_classes"] = disease_types  # e.g., ["AD", "DLB", "HV", "NPH"]
        dataset_str = BIDS_ROOT.split("/")[-1]

        # Try to load from the cache file
        cache_path = f"./tmp/data_all_{dataset_str}.pkl"
        if not from_pkl:
            data_all = loader_func(load_params)  # takes time
            mngs.io.save(data_all, cache_path)

        else:
            try:
                print(f"\n\n\n !!! Loading from a cache file ({cache_path}) !!! \n\n\n")
                data_all = mngs.io.load(cache_path)
            except Exception as e:
                print(e)
                data_all = loader_func(load_params)
                mngs.io.save(data_all, cache_path)

        ## drops not-labeled? data
        expanded_classes = np.hstack(_expand_conc_classes_str(disease_types))
        indi_to_use = (
            pd.concat(
                [data_all["disease_type"] == ec for ec in expanded_classes], axis=1
            )
            .sum(axis=1)
            .astype(bool)
        )
        data_not_used = data_all.loc[~indi_to_use]
        data_all = data_all.loc[indi_to_use]
        return load_params, data_all, data_not_used

    load_params, data_all, data_not_used = load_BIDS("./data/BIDS_Osaka", from_pkl=True)
    div_params = load_params
    div_params["val_ratio_to_trn"] = 0
    div_params["num_folds"] = 1
    div_dict = divide_into_datasets(data_all, div_params)
