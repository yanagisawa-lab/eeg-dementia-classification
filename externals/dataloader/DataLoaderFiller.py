#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import sys
import warnings
from copy import deepcopy
from functools import partial
from pprint import pprint
from time import sleep

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import sklearn
import torch
from natsort import natsorted
try:
    from pandas.core.common import SettingWithCopyWarning # 1.4
except:
    from pandas.errors import SettingWithCopyWarning # 1.5
from sklearn.preprocessing import power_transform
from torch.utils.data import DataLoader, Dataset

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

sys.path.append("./externals")
from dataloader.load_BIDS import load_BIDS as load_BIDS_func
from dataloader.utils.divide_into_datasets import divide_into_datasets
from dataloader.utils.label_handlers import _expand_conc_classes_str


class DataLoaderFiller(object):
    """
    Fills cropped EEG samples and their labels into self.dl_tra, self.dl_val, and self.dl_tes,
    each of which is torch.utils.data.DataLoader. These data loaders yields batches,
    and every batch contains the followings in this order:

        Xb: batched inputs, EEG (n_samples * n_channel * seq_len)
        Tb: batched targets, Diagnosed labels
        Sgb: batched Subjects ID (global)
        Slb: batched Subjects ID, reallocated within a dataset like Training Dataset (local)

    Examples:

    # ----------------------------------------
    # NN Training
    # ----------------------------------------
    BIDS_ROOT = (
        f"/storage/dataset/EEG/internal/EEG_DiagnosisFromRawSignals/"
        f"BIDS_dataset_v5.2_Osk"
    )
    disease_types=["HV", "AD", "NPH", "DLB"]

    dlf = DataLoaderFiller(BIDS_ROOT, disease_types)

    n_epochs = 3
    for i_fold in range(dlf.filler_params["num_folds"]):
        dlf.fill(i_fold, reset_fill_counter=True)
        for epoch in range(n_epochs):
            dlf.fill(i_fold, reset_fill_counter=False)

            for i_batch, batch in enumerate(dlf.dl_val):
                Xb, Tb, Sgb, Slb = batch
                # Validate your model

            for i_batch, batch in enumerate(dlf.dl_tra):
                Xb, Tb, Sgb, Slb = batch
                # Train your model

        for i_batch, batch in enumerate(dlf.dl_tes):
            Xb, Tb, Sgb, Slb = batch
            # Test your model

    # Summarize Cross Validation Scores

    # ----------------------------------------
    # Makes a dataloader from an existing BIDS layout
    # ----------------------------------------
    BIDS_ROOT = (
        f"/storage/dataset/EEG/internal/EEG_DiagnosisFromRawSignals/"
        f"BIDS_dataset_v5.2_Osk"
    )
    load_params, data_all, data_not_used = DataLoaderFiller.load_BIDS(
        BIDS_ROOT,
        disease_types=["HV", "AD", "NPH", "DLB"],
        from_pkl=True,
        samp_rate=500,
    )
    subs_labels_uq = data_all[["subject", "disease_type"]].drop_duplicates
    window_size_pts = 1000
    sub_str2int_local = {s:i for i,s in enumerate(subs_labels_uq["subject"])}
    sub_str2int_global = sub_str2int_local

    dl = DataLoaderFiller.crop_without_overlap(
        data_all,
        subs_labels_uq["subject"],
        subs_labels_uq["disease_type"],
        window_size_pts,
        sub_str2int_val,
        sub_str2int_global,
        crop_num=None,
        dtype_np=np.float32,
        transform=None,
        field_order=None,
        batch_size=32,
        num_workers=16,
    )

    # ----------------------------------------
    # Directly accesses the dataset
    # ----------------------------------------
    X_tra, T_tra, Sg_tra, Sl_tra = dlf.dl_tra.dataset.arrs_list
    X_val, T_val, Sg_val, Sl_val = dlf.dl_val.dataset.arrs_list
    X_tes, T_tes, Sg_tes, Sl_tes = dlf.dl_tes.dataset.arrs_list

    Note:
        dir(dlf) or dlf.[TAB KEY] will show you availabel attributes and methods.
        Especially,

    """

    def __init__(
        self,
        BIDS_ROOT,
        disease_types,
        stop_bands=None,
        val_ratio_to_trn=None,
        is_no_test=None,
        drop_cMCI=False,
        window_size_pts=1000,
        montage=None,
    ):

        self.load_params, self.data_all, self.data_not_used = self.load_BIDS(
            BIDS_ROOT,
            disease_types=disease_types,
            samp_rate=500,
            drop_cMCI=drop_cMCI,
            montage=montage,
        )

        # self.data_all = self._process_with_demographic_data(self.data_all)

        if val_ratio_to_trn is not None:
            self.load_params["val_ratio_to_trn"] = val_ratio_to_trn  # override

        if is_no_test is True:
            self.load_params["num_folds"] = 1  # override

        if stop_bands is not None:
            for band_str in stop_bands:
                self.data_all = self._band_stop(self.data_all, band_str)

        self.filler_params = self._load_and_process_filler_params(
            self.load_params["conc_classes"]
        )
        self.filler_params["window_size_pts"] = window_size_pts
        self.cv_dict = self._mk_cv_dict()

        self.db = self._mk_db(self.data_all, self.cv_dict)
        self._fix_seeds(random_state=self.filler_params["random_state"])
        self._fill_counter = 0

    def _band_stop(self, data_all, band_str):
        def _inner_band_stop(low_hz, high_hz, eeg_record):
            return np.array(
                mngs.dsp.bandstop(
                    torch.tensor(eeg_record).unsqueeze(0).cuda(),
                    samp_rate=500,
                    low_hz=low_hz,
                    high_hz=high_hz,
                ).cpu()
            ).squeeze()

        BANDS_LIM_HZ = mngs.io.load("./config/global.yaml")["BANDS_LIM_HZ_DICT"]
        low_hz, high_hz = BANDS_LIM_HZ[band_str]

        p_inner_band_stop = partial(_inner_band_stop, low_hz, high_hz)

        print(f"\n{band_str} band is being suppressed ...\n")
        self.data_all["eeg"] = self.data_all["eeg"].apply(p_inner_band_stop)

        return self.data_all

    def _process_data_all(self, data_all):
        load_params["conc_classes"] = disease_types  # e.g., ["AD", "DLB", "HV", "NPH"]
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

    def _mk_cv_dict(
        self,
    ):
        div_dict = {}
        keys = [
            "class_field_name",
            "num_folds",
            "conc_classes",
            "val_ratio_to_trn",
            "undersample_trn",
            "undersample_val",
            "random_state",
        ]

        for k in keys:
            try:
                div_dict[k] = self.filler_params[k]
            except:
                div_dict[k] = self.load_params[k]

        cv_dict = divide_into_datasets(self.data_all, div_dict)

        return cv_dict

    def fill(self, i_fold, reset_fill_counter=True):
        """
        Fills self.dl_tra, self.dl_val, and self.dl_tes.

        self.dl_val and self.dl_tes are always the identical ones because they have no randomness.
        """
        self.i_fold = i_fold

        if reset_fill_counter:
            self._fill_counter = 0
            self._check_cv_health(self.cv_dict, i_fold)

        self.dl_tra = self._fill_twice_length_trn_ds(
            self.data_all,
            self.cv_dict["label_int_2_conc_class_dict"],
            self.subs_trn,
            self.labels_trn,
            self.sub_str2int_trn,
            self.sub_str2int_global,
            self.filler_params["window_size_pts"],
            dtype_np=self.filler_params["_dtype_np"],
            field_order=self.filler_params["field_order"],
            batch_size=self.filler_params["batch_size"],
            num_workers=self.filler_params["num_workers"],
        )

        if self._fill_counter == 0:

            if self.load_params["val_ratio_to_trn"] == 0:
                self._dl_val = None
            else:
                self._dl_val = self.crop_without_overlap(
                    self.data_all,
                    self.subs_val,
                    self.labels_val,
                    self.filler_params["window_size_pts"],
                    self.sub_str2int_val,
                    self.sub_str2int_global,
                    crop_num=None,
                    dtype_np=self.filler_params["_dtype_np"],
                    transform=self.filler_params["transform"],
                    field_order=self.filler_params["field_order"],
                    batch_size=self.filler_params["batch_size"],
                    num_workers=self.filler_params["num_workers"],
                )

            self._dl_tes = self.crop_without_overlap(
                self.data_all,
                self.subs_tes,
                self.labels_tes,
                self.filler_params["window_size_pts"],
                self.sub_str2int_tes,
                self.sub_str2int_global,
                crop_num=None,
                dtype_np=self.filler_params["_dtype_np"],
                transform=self.filler_params["transform"],
                field_order=self.filler_params["field_order"],
                batch_size=self.filler_params["batch_size"],
                num_workers=self.filler_params["num_workers"],
            )

            self.dl_val = deepcopy(self._dl_val)
            self.dl_tes = deepcopy(self._dl_tes)

        if self._fill_counter >= 1:
            self.dl_val = deepcopy(self._dl_val)
            self.dl_tes = deepcopy(self._dl_tes)

        self._fill_counter += 1

    def _fill_twice_length_trn_ds(
        self,
        data_all,
        label2class_dict,
        subs_trn,
        labels_trn,
        subj_str2int_trn,
        subj_str2int_global,
        window_size_pts,
        dtype_np=np.float32,
        transform=None,
        field_order=None,
        batch_size=64,
        num_workers=1,
    ):
        """
        After once an twice length Training Dataset was prepared with self._crop_trn_random(),
        this function fills such a dataset into a data loader with perturbation,
        which is also called as sliding-window data augumentation. This is rendered by
        RandomCropepr passed as a collate_fn, leading batched EEG samples to be cropped
        to their half, target length.
        """
        if self._fill_counter == 0:
            # Creates a twice-long training dataset

            # Determines the numbers to sample from each subject
            samps_trn_all = []
            for sub in subs_trn:
                EEGs_sub = data_all["eeg"][data_all["subject"] == sub]
                _n_samps = [
                    EEGs_sub.iloc[i_row].shape[1] // window_size_pts
                    for i_row in range(len(EEGs_sub))
                ]
                samps = np.sum(_n_samps)
                samps_trn_all.append(samps)
            n_trn_subjs = len(samps_trn_all)
            n_95_percentile = sorted(samps_trn_all)[int(n_trn_subjs * 0.05)]
            print(
                f"\n{n_95_percentile} x EEG segments will be sampled from each subject.\n"
            )

            # Makes Training Dataset, whose EEG samples have twice long duration
            self._ds_tra_twice_long = self._crop_trn_random(
                data_all,
                label2class_dict,
                subs_trn,
                labels_trn,
                n_95_percentile,
                subj_str2int_trn,
                subj_str2int_global,
                2 * window_size_pts,
                dtype_np=dtype_np,
                field_order=field_order,
            )

        RC = RandomCropper(window_size_pts, random_state=self._fill_counter)

        dl_tra = DataLoader(
            self._ds_tra_twice_long,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            worker_init_fn=seed_worker,
            pin_memory=True,
            collate_fn=RC.random_crop,
        )
        return dl_tra

    @staticmethod
    def _crop_trn_random(
        data_all,
        label2class_dict,
        subs_trn,
        labels_trn,
        max_n_EEGs_per_sub,
        subj_str2int_trn,
        subj_str2int_global,
        window_size_pts,
        dtype_np=np.float32,
        transform=None,
        field_order=None,
    ):
        def _crop_signal_sub_random(data_all, sub, crop_len, crop_num, dtype):

            EEGs_sub = data_all["eeg"][data_all["subject"] == sub]
            sample_nums = np.array([x.shape[1] for x in EEGs_sub])
            start_cand = sample_nums - crop_len + 1
            start_cand[start_cand < 0] = 0
            start_cumsum = np.cumsum(start_cand)

            EEG_cropped = []
            for crop_i in range(crop_num):
                start_i = np.random.randint(0, np.sum(start_cand), 1)[0]
                rec_i = np.nonzero(start_i < start_cumsum)
                rec_i = rec_i[0][0]

                if rec_i == 0:
                    data_i = start_i
                else:
                    data_i = start_i - start_cumsum[rec_i - 1]
                tmp_signal = EEGs_sub.values[rec_i][:, data_i : (data_i + crop_len)]
                tmp_signal = tmp_signal[np.newaxis, :, :]  # sample (1) * channel * time

                EEG_cropped.append(tmp_signal)

            EEG_cropped = np.concatenate(EEG_cropped, axis=0).astype(dtype)

            return EEG_cropped

        label_num = len(label2class_dict)
        subs_trn_by_classes = [None for _ in range(label_num)]
        for label_i in range(label_num):
            subs_trn_by_classes[label_i] = [
                x for (i, x) in enumerate(subs_trn) if labels_trn[i] == label_i
            ]
        subs_num_trn_by_classes = np.array([len(x) for x in subs_trn_by_classes])
        tot_crop_num_from_a_class = max_n_EEGs_per_sub * np.min(subs_num_trn_by_classes)

        crop_num_trn_by_classes = [None for _ in range(label_num)]
        for label_i in range(label_num):
            div_num = tot_crop_num_from_a_class // subs_num_trn_by_classes[label_i]
            mod_num = tot_crop_num_from_a_class % subs_num_trn_by_classes[label_i]

            crop_num_trn_by_classes[label_i] = (
                np.ones(subs_num_trn_by_classes[label_i]) * div_num
            )

            crop_num_trn_by_classes[label_i][0:mod_num] = (
                crop_num_trn_by_classes[label_i][0:mod_num] + 1
            )

        _data = data_all.set_index("subject").loc[np.hstack(subs_trn_by_classes)]
        _data_uq = _data[~_data.index.duplicated()]

        # fill nan on age, sex, and MMSE by their median values
        _data_uq = fill_age(_data_uq, "age")
        _data_uq = fill_sex(_data_uq, "sex")
        _data_uq = fill_MMSE(_data_uq, "MMSE")

        # normalizing age, sex, and MMSE
        _data_uq["age_normed"] = transform_sex(_data_uq["age"])
        _data_uq["sex_normed"] = transform_sex(_data_uq["sex"])
        _data_uq["MMSE_normed"] = transform_MMSE(_data_uq["MMSE"])
        _data_uq = _data_uq.reset_index()

        X, T, Sg, Sl, A, G, M = [], [], [], [], [], [], []
        for class_i, subs in enumerate(subs_trn_by_classes):
            crop_nums = deepcopy(crop_num_trn_by_classes[class_i])
            np.random.shuffle(crop_nums)
            for sub_i, sub in enumerate(subs):
                EEG_cropped = _crop_signal_sub_random(
                    data_all,
                    sub,
                    window_size_pts,
                    int(crop_nums[sub_i]),
                    dtype_np,
                )
                X.append(EEG_cropped)
                T.append(np.ones((EEG_cropped.shape[0])) * class_i)
                Sg.append(np.ones((EEG_cropped.shape[0])) * subj_str2int_global[sub])
                Sl.append(np.ones((EEG_cropped.shape[0])) * subj_str2int_trn[sub])
                A.append(
                    np.ones((EEG_cropped.shape[0]))
                    * _data_uq[_data_uq["subject"] == sub]["age_normed"].iloc[0]
                )
                G.append(
                    np.ones((EEG_cropped.shape[0]))
                    * _data_uq[_data_uq["subject"] == sub]["sex_normed"].iloc[0]
                )
                M.append(
                    np.ones((EEG_cropped.shape[0]))
                    * _data_uq[_data_uq["subject"] == sub]["MMSE_normed"].iloc[0]
                )

        X = np.concatenate(X, axis=0)
        T = np.concatenate(T, axis=0).astype(np.int32).tolist()
        Sg = np.concatenate(Sg, axis=0).astype(np.int32).tolist()
        Sl = np.concatenate(Sl, axis=0).astype(np.int32).tolist()
        A = np.concatenate(A, axis=0).astype(np.float32).tolist()
        G = np.concatenate(G, axis=0).astype(np.int32).tolist()
        M = np.concatenate(M, axis=0).astype(np.float32).tolist()

        arrs_list_to_pack = sklearn.utils.shuffle(X, T, Sg, Sl, A, G, M)

        assert np.all([len(arr) for arr in arrs_list_to_pack])
        if field_order is not None:
            arrs_list_to_pack = tuple([arrs_list_to_pack[x] for x in field_order])

        ds = _CustomDataset(arrs_list_to_pack, transform=transform)

        return ds

    @classmethod
    def crop_without_overlap(
        cls,
        data_all,
        subs,
        labels,
        crop_len,
        subj_str2int_local,
        subj_str2int_global,
        dtype_np=np.float32,
        crop_num=None,
        transform=None,
        field_order=None,
        batch_size=64,
        num_workers=1,
    ):
        def _crop_signal_sub_without_overlap(data_all, sub, crop_len, crop_num, dtype):
            EEG_cropped = []
            EEGs_sub = data_all["eeg"][data_all["subject"] == sub]
            for EEG in EEGs_sub:

                sample_count = EEG.shape[1] // crop_len
                tmp_signal = EEG[:, 0 : (crop_len * sample_count)]
                tmp_signal = np.reshape(
                    tmp_signal, (tmp_signal.shape[0], crop_len, sample_count), order="F"
                )
                tmp_signal = np.moveaxis(tmp_signal, -1, 0)  # sample * channel * time
                EEG_cropped.append(tmp_signal)

            EEG_cropped = np.concatenate(EEG_cropped, axis=0).astype(dtype)

            if (crop_num is not None) and (EEG_cropped.shape[0] > crop_num):
                EEG_cropped = EEG_cropped[0:crop_num, :, :]

            return EEG_cropped

        _data = data_all.set_index("subject").loc[np.hstack(subs)]
        _data_uq = _data[~_data.index.duplicated()]

        # fill nan on age, sex, and MMSE by their median values
        _data_uq = fill_age(_data_uq, "age")
        _data_uq = fill_sex(_data_uq, "sex")
        _data_uq = fill_MMSE(_data_uq, "MMSE")

        # normalizing age, sex, and MMSE
        _data_uq["age_normed"] = transform_age(_data_uq["age"])
        _data_uq["sex_normed"] = transform_sex(_data_uq["sex"])
        _data_uq["MMSE_normed"] = transform_MMSE(_data_uq["MMSE"])
        _data_uq = _data_uq.reset_index()

        X, T, Sg, Sl, A, G, M = [], [], [], [], [], [], []
        for sub_i, sub in enumerate(subs):
            EEG_cropped = _crop_signal_sub_without_overlap(
                data_all, sub, crop_len, crop_num, dtype_np
            )
            X.append(EEG_cropped)
            T.append(np.ones((EEG_cropped.shape[0])) * labels[sub_i])
            Sg.append(np.ones((EEG_cropped.shape[0])) * subj_str2int_global[sub])
            Sl.append(np.ones((EEG_cropped.shape[0])) * subj_str2int_local[sub])
            A.append(
                np.ones((EEG_cropped.shape[0]))
                * _data_uq[_data_uq["subject"] == sub]["age_normed"].iloc[0]
            )
            G.append(
                np.ones((EEG_cropped.shape[0]))
                * _data_uq[_data_uq["subject"] == sub]["sex_normed"].iloc[0]
            )
            M.append(
                np.ones((EEG_cropped.shape[0]))
                * _data_uq[_data_uq["subject"] == sub]["MMSE_normed"].iloc[0]
            )

        assert (
            len(subj_str2int_local)
            == len(np.unique(np.hstack(Sg)))
            == len(np.unique(np.hstack(Sl)))
        )
        assert set(subj_str2int_local.values()) == set(np.unique(np.hstack(Sl)))

        X = np.concatenate(X, axis=0)
        T = np.concatenate(T, axis=0).astype(np.int32).tolist()
        Sg = np.concatenate(Sg, axis=0).astype(np.int32).tolist()
        Sl = np.concatenate(Sl, axis=0).astype(np.int32).tolist()
        A = np.concatenate(A, axis=0).astype(np.float32).tolist()
        G = np.concatenate(G, axis=0).astype(np.int32).tolist()
        M = np.concatenate(M, axis=0).astype(np.float32).tolist()

        arrs_list_to_pack = sklearn.utils.shuffle(X, T, Sg, Sl, A, G, M)
        assert np.all([len(arr) for arr in arrs_list_to_pack])
        if field_order is not None:
            arrs_list_to_pack = tuple([arrs_list_to_pack[x] for x in field_order])

        ds = _CustomDataset(
            arrs_list_to_pack,
            transform=transform,
            order=field_order,
        )
        # Please allow "shuffle=True" to estimate validation metrics from only batches
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            worker_init_fn=seed_worker,
            pin_memory=True,
        )
        return dl

    @staticmethod
    def _fix_seeds(random_state=42):
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def _load_and_process_filler_params(conc_classes):
        # loads filler_params
        filler_params = mngs.io.load("./config/filler_params.yaml")
        filler_params["conc_classes"] = conc_classes
        filler_params.setdefault("class_field_name", "disease_type")
        # filler_params.setdefault("window_size_pts", 1000)
        filler_params.setdefault("random_state", 42)
        filler_params.setdefault("batch_size", 64)
        filler_params.setdefault("num_workers", 16)
        filler_params.setdefault("dtype", "fp32")
        filler_params.setdefault("field_order", [0, 1, 2, 3, 4, 5, 6])
        filler_params.setdefault("transform", None)
        filler_params.setdefault("is_debug_mode", False)

        # additional params
        filler_params["_dtype_np"] = (
            np.float32 if filler_params["dtype"] == "fp32" else np.float16
        )

        return filler_params

    @classmethod
    def _check_cv_health(cls, cv_dict, i_fold):
        assert len(set(cv_dict["subs_trn"][i_fold])) == len(cv_dict["subs_trn"][i_fold])
        assert len(set(cv_dict["subs_val"][i_fold])) == len(cv_dict["subs_val"][i_fold])
        assert len(set(cv_dict["subs_tes"][i_fold])) == len(cv_dict["subs_tes"][i_fold])

        assert (
            len(set(cv_dict["subs_trn"][i_fold]) & set(cv_dict["subs_val"][i_fold]))
            == 0
        )
        assert (
            len(set(cv_dict["subs_val"][i_fold]) & set(cv_dict["subs_tes"][i_fold]))
            == 0
        )
        assert (
            len(set(cv_dict["subs_tes"][i_fold]) & set(cv_dict["subs_trn"][i_fold]))
            == 0
        )
        print("\nCross Validation dict is healthy.\n")

    @staticmethod
    def _mk_db(data_all, cv_dict):
        """Makes a global database."""
        db = pd.DataFrame()
        db = (
            data_all[
                ["subject", "disease_type", "cognitive_level", "age", "sex", "MMSE"]
            ]
            .drop_duplicates()
            .set_index("subject")
        )
        db = db.rename(columns={"disease_type": "disease_type_str"})
        disease_type_int = []

        for ii in range(len(db)):
            try:
                disease_type_int.append(
                    cv_dict["conc_class_str_2_label_int_dict"][
                        db["disease_type_str"].iloc[ii]
                    ]
                )
            except Exception as e:
                print(e)
                disease_type_int.append(-1)

        db["disease_type_int"] = disease_type_int
        db["global_id"] = np.arange(len(db))

        for i_fold, (subs_trn, subs_val, subs_tes) in enumerate(
            zip(cv_dict["subs_trn"], cv_dict["subs_val"], cv_dict["subs_tes"])
        ):
            db.loc[subs_trn, f"fold#{i_fold}_dataset"] = "Training"
            db.loc[subs_trn, f"fold#{i_fold}_local_id"] = np.arange(
                len(subs_trn), dtype=int
            )

            db.loc[subs_val, f"fold#{i_fold}_dataset"] = "Validation"
            db.loc[subs_val, f"fold#{i_fold}_local_id"] = np.arange(
                len(subs_val), dtype=int
            )

            db.loc[subs_tes, f"fold#{i_fold}_dataset"] = "Test"
            db.loc[subs_tes, f"fold#{i_fold}_local_id"] = np.arange(
                len(subs_tes), dtype=int
            )
        return db

    @classmethod
    def load_BIDS(
        cls,
        BIDS_ROOT,
        disease_types=["HV", "AD", "NPH", "DLB"],
        samp_rate=500,
        drop_cMCI=False,
        montage=None,
    ):

        load_params = mngs.io.load("./config/load_params.yaml")

        if montage != None:
            load_params["montage"] = montage

        load_params["BIDS_root"] = BIDS_ROOT
        load_params["conc_classes"] = disease_types  # e.g., ["AD", "DLB", "HV", "NPH"]
        dataset_str = BIDS_ROOT.split("/")[-1]

        data_all = load_BIDS_func(load_params)                

        ## drops not-labeled? data
        expanded_classes = np.hstack(_expand_conc_classes_str(disease_types))
        indi_to_use = (
            pd.concat(
                [data_all["disease_type"] == ec for ec in expanded_classes], axis=1
            )
            .sum(axis=1)
            .astype(bool)
        )
        if drop_cMCI:
            indi_drop = data_all["cognitive_level"] == "cMCI"
            indi_to_use = indi_to_use * ~indi_drop

        data_not_used = data_all.loc[~indi_to_use]
        data_all = data_all.loc[indi_to_use]

        return load_params, data_all, data_not_used

    @property
    def sex_str2int_dict(self):
        return {"M": 0, "F": 1}

    @property
    def sex_int2str_dict(self):
        return {v: k for k, v in self.sex_str2int_dict.items()}

    @property
    def sub_str2int_global(
        self,
    ):
        return {s: i for s, i in zip(self.db.index, self.db["global_id"])}

    @property
    def sub_int2str_global(
        self,
    ):
        return {i: s for s, i in self.sub_str2int_global.items()}

    def _sub_str2int_local(self, dataset_str):
        _db = self.db[self.db[f"fold#{self.i_fold}_dataset"] == dataset_str][
            f"fold#{self.i_fold}_local_id"
        ]
        return {s: int(i) for s, i in zip(_db.index, _db)}

    def _sub_int2str_local(self, dataset_str):
        return {i: s for s, i in self._sub_str2int_local(dataset_str).items()}

    @property
    def sub_str2int_trn(
        self,
    ):
        return self._sub_str2int_local("Training")

    @property
    def sub_int2str_trn(
        self,
    ):
        return self._sub_int2str_local("Training")

    @property
    def sub_str2int_val(
        self,
    ):
        return self._sub_str2int_local("Validation")

    @property
    def sub_int2str_val(
        self,
    ):
        return self._sub_int2str_local("Validation")

    @property
    def sub_str2int_tes(
        self,
    ):
        return self._sub_str2int_local("Test")

    @property
    def sub_int2str_tes(
        self,
    ):
        return self._sub_int2str_local("Test")

    @property
    def subs_all(
        self,
    ):
        return np.array(
            natsorted(np.hstack([self.subs_trn, self.subs_val, self.subs_tes]))
        )

    @property
    def subs_trn(
        self,
    ):
        return list(self.sub_int2str_trn.values())

    @property
    def subs_val(
        self,
    ):
        return list(self.sub_int2str_val.values())

    @property
    def subs_tes(
        self,
    ):
        return list(self.sub_int2str_tes.values())

    def _labels_dataset(self, dataset_str):
        return np.array(
            self.db["disease_type_int"][
                self.db[f"fold#{self.i_fold}_dataset"] == dataset_str
            ]
        )

    @property
    def labels_trn(
        self,
    ):
        return self._labels_dataset("Training")

    @property
    def labels_val(
        self,
    ):
        return self._labels_dataset("Validation")

    @property
    def labels_tes(
        self,
    ):
        return self._labels_dataset("Test")

    @classmethod
    def transform_age(cls, x):
        return transform_age(x)

    @classmethod
    def transform_sex(cls, x):
        return transform_sex(x)

    @classmethod
    def transform_MMSE(cls, x):
        return transform_MMSE(x)


class RandomCropper:
    def __init__(self, window_size_pts, random_state=42):
        self.window_size_pts = window_size_pts
        # self.random_state = random_state
        self.rs = np.random.RandomState(random_state)

    def random_crop(self, data_list):
        """
        len(data_list) # batch_size
        tensors_list = data_list[0]
        len(tensors_list) # 4 # Xb, Tb, Sgb, Slb

        A transform function on batched EEG segments.
        Note: X_arr.shape -> [n_chs, window_size_pts]
        """

        n_elements = len(data_list[0])

        elements = []
        for i_elements in range(n_elements):
            elements.append(torch.stack([tl[i_elements] for tl in data_list]))

        # Randomly crop EEG segments
        Xb = elements[0]
        rand_start = self.rs.randint(Xb.shape[-1] - (self.window_size_pts - 1))
        rand_end = rand_start + self.window_size_pts
        Xb = Xb[..., rand_start:rand_end]
        elements[0] = Xb

        # print(self.random_state)

        return elements


class _CustomDataset(Dataset):
    """
    Example:
        n = 1024
        n_chs = 19
        X = np.random.rand(n, n_chs, 1000)
        T = np.random.randint(0, 4, size=(n, 1))
        S = np.random.randint(0, 999, size=(n, 1))
        Sl = np.random.randint(0, 4, size=(n, 1))

        arrs_list = [X, T, S, Sl]
        transform = None
        ds = _CustomDataset(arrs_list, transform=transform)
        len(ds) # 1024
    """

    def __init__(self, arrs_list, transform=None, order=[0, 1, 2, 3, 4, 5, 6]):
        """
        Arguments:
            'transform'
                a function for transforming X (= EEG) like applying data augmentation.
                Pseudo-code is something like below.
                X_arr = transform(X_arr) # X_arr.shape [n_chs, window_size_pts]
        """

        self.arrs_list = arrs_list
        assert np.all([len(arr) for arr in arrs_list])
        self.length = len(arrs_list[0])
        self.transform = transform
        self.order = order

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        r_list = [l[idx] for l in self.arrs_list]

        if self.transform:
            dtype_orig = r_list[0].dtype
            r_list[self.order.index(0)] = self.transform(
                r_list[0].astype(np.float64)
            ).astype(dtype_orig)

        return [torch.tensor(d) for d in r_list]


def seed_worker(worker_id):
    """
    https://dajiro.com/entry/2021/04/13/233032
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def transform_age(x):
    x = x.copy()
    return (x - x.mean()) / x.std()


def transform_sex(x):
    x = x.copy()
    return x


def transform_MMSE(x):
    x = x.copy()
    minus_MMSE = np.array(31 - x)
    log_minus_MMSE = np.log(minus_MMSE + 1e-5)
    norm_log_minus_MMSE = (
        log_minus_MMSE - log_minus_MMSE.mean()
    ) / log_minus_MMSE.std()
    return norm_log_minus_MMSE


def fill_age(df, key):
    return _fill_by_median(df, key)


def fill_MMSE(df, key):
    return _fill_by_median(df, key)


def _fill_by_median(df, key):
    col = df[key].copy()
    median = np.nanmedian(col)
    df.loc[col.isna(), key] = median
    return df


def fill_sex(df, key):
    col = df[key].copy()
    val = 0.5
    df.loc[col.isna(), key] = val
    return df


if __name__ == "__main__":
    disease_types = ["HV", "AD", "NPH", "DLB"]
    dlf = DataLoaderFiller(
        "./data/BIDS_Kochi",
        disease_types,
        drop_cMCI=True,
    )
    dlf.fill(i_fold=0, reset_fill_counter=True)

    """    
    SG_tra = dlf.dl_tra.dataset.arrs_list[2]
    SG_val = dlf.dl_val.dataset.arrs_list[2]
    SG_tes = dlf.dl_tes.dataset.arrs_list[2]
    len(np.unique(SG_tra + SG_val + SG_tes)) # 204

    T_tra = dlf.dl_tra.dataset.arrs_list[1]
    T_val = dlf.dl_val.dataset.arrs_list[1]
    T_tes = dlf.dl_tes.dataset.arrs_list[1]
    np.unique(T_tra, return_counts=True)
    np.unique(np.array(T_tra + T_val + T_tes), return_counts=True)
    
    """

    n_epochs = 3
    for i_fold in range(dlf.load_params["num_folds"]):
        dlf.fill(i_fold, reset_fill_counter=True)
        for epoch in range(n_epochs):
            dlf.fill(i_fold, reset_fill_counter=False)

            # Training dataset
            for i_batch, batch in enumerate(dlf.dl_tra):
                Xb_tra, Tb_tra, Sgb_tra, Slb_tra, A_tra, G_tra, M_tra = batch
                print(Xb_tra)

            # Validation dataset
            for i_batch, batch in enumerate(dlf.dl_val):
                Xb_val, Tb_val, Sgb_val, Slb_val, A_val, G_val, M_val = batch

        # Test dataset
        for i_batch, batch in enumerate(dlf.dl_tra):
            Xb_tra, Tb_tra, Sgb_tra, Slb_tra, A_tra, G_tra, M_tra = batch
