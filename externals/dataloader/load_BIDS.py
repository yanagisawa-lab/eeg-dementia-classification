#!/usr/bin/env python3

import itertools
import re
import sys
import warnings
from copy import deepcopy

import bids
import mngs
import numpy as np
import pandas as pd
from bids import BIDSLayout
from mne.io import read_raw_edf
from natsort import natsorted

try:
    from pandas.core.common import SettingWithCopyWarning  # 1.4
except:
    from pandas.errors import SettingWithCopyWarning

from scipy import stats
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

sys.path.append("./externals")

import datetime

from dataloader.signal_filters import BandPassFilter, NotchFilter
from scipy.signal import resample


def load_BIDS(load_params, verbose=False, drop_unrelated=True):
    data_all = _load_BIDS(load_params)
    if drop_unrelated:
        data_all = _drop_data(data_all)
    return data_all


def _load_BIDS(load_params, verbose=False):
    """
    Arguments:

        load_params

            ['BIDS_root']
            ['subjects']
            ['montage']
            ['trial_types']
            ['load_signal']
            ['apply_common-average']

    Returns:
        pandas.DataFrame object
    """

    # Defaults prameters
    load_params.setdefault("subjects", None)
    load_params.setdefault("load_signal", True)
    load_params.setdefault("apply_common-average", True)

    # Determins how to take references
    if load_params["load_signal"]:
        assert "montage" in load_params.keys()
        assert type(load_params["montage"]) is list
        assert len(load_params["montage"]) > 0

        is_monopolar = type(load_params["montage"][0]) is str
        if is_monopolar:
            assert np.all(np.array([type(x) is str for x in load_params["montage"]]))
            load_params["channels"] = load_params["montage"]
            load_params["is_unipolar"] = True
        else:
            assert np.all(np.array([type(x) is list for x in load_params["montage"]]))
            assert np.all(np.array([len(x) for x in load_params["montage"]]) == 2)
            assert np.all(
                np.array(
                    [
                        np.all(np.array([type(y) is str for y in x]))
                        for x in load_params["montage"]
                    ]
                )
            )
            load_params["channels"] = list(
                set(itertools.chain.from_iterable(load_params["montage"]))
            )
            load_params["is_unipolar"] = False
            load_params["montage2channels"] = np.stack(
                [
                    np.array([load_params["channels"].index(y) for y in x])
                    for x in load_params["montage"]
                ],
                axis=1,
            ).tolist()



    # Loads a BIDS dataset        
    layout = BIDSLayout(load_params["BIDS_root"])
    entities = layout.get_entities()

    if verbose:
        print("\nLoading BIDS dataset ...\n")
        print(f"\nChannels:\n{load_params['channels']}\n")
        
        layout_keys_to_print = [
            "InstitutionName",
            "PowerLineFrequency",
            "SamplingFrequency",
            "TaskName",
            "extension",
            "run",
            "session",
            "subject",
            "suffix",
            "task",
        ]

        for k in layout_keys_to_print:
            print(f"\nunique {k}s:\n{natsorted(entities[k].unique())}\n")
            sleep(1)

    subj_uq = natsorted(entities["subject"].unique())
    task_uq = natsorted(entities["task"].unique())
    datatype_uq = natsorted(entities["datatype"].unique())
    session_uq = natsorted(entities["session"].unique())
    run_uq = natsorted(entities["run"].unique())

    data_all = []
    for i_subj, subj in tqdm(enumerate(subj_uq)):

        try:
            if (load_params["subjects"] is not None) and (
                subj not in load_params["subjects"]
            ):
                continue

            # Session
            session_bids = layout.get(subject=subj, suffix="sessions")
            assert len(session_bids) == 1
            session_bids = session_bids[0]

            for task in task_uq:
                for datatype in datatype_uq:
                    for session in session_uq:
                        for run in run_uq:

                            ## EDF, signal
                            edf_bids = layout.get(
                                subject=subj,
                                session=session,
                                run=run,
                                datatype=datatype,
                                extension=".edf",
                            )
                            assert len(edf_bids) == 1
                            edf_bids = edf_bids[0]

                            ## Event
                            event_bids = layout.get(
                                subject=subj,
                                session=session,
                                run=run,
                                datatype=datatype,
                                suffix="events",
                                extension=".tsv",
                            )
                            assert len(event_bids) == 1
                            event_bids = event_bids[0]

                            ## eeg metadata from json
                            metadata_bids = layout.get(
                                subject=subj,
                                session=session,
                                run=run,
                                datatype=datatype,
                                suffix="eeg",
                                extension=".json",
                            )
                            assert len(metadata_bids) == 1
                            metadata_bids = metadata_bids[0]

                            ## Load data for each run
                            data_run = _load_a_run(
                                session_bids,
                                edf_bids,
                                event_bids,
                                metadata_bids,
                                load_params,
                            )

                            ## Buffering
                            data_all.append(data_run)

        except Exception as e:
            print(f"\n{subj}:\n{e}\n")

    data_all_df = pd.concat(data_all).reset_index()
    del data_all_df["index"]

    data_all_df = _add_age_sex_and_MMSE(data_all_df, load_params["BIDS_root"])

    data_all_df = _to_numeric(data_all_df)
    # data_all_df = _mask_age_sex_and_MMSE(data_all_df, load_params)
    data_all_df = _rename_cols(data_all_df)

    return data_all_df


def _load_a_run_wrapper(args_list):
    return _load_a_run(*args_list)


def _load_a_run(
    session_bids, edf_bids, event_bids, metadata_bids, load_params, verbose=False
):
    event_df = event_bids.get_df()
    event_df["offset"] = event_df["onset"] + event_df["duration"]
    metadata_dict = metadata_bids.get_dict()

    ## Excludes unnecessary periods depending on the keyward argument trial_types
    event_df = pd.concat(
        [event_df[event_df["trial_type"] == utt] for utt in load_params["trial_types"]]
    ).sort_values("onset")

    df = event_df.copy()

    if load_params["load_signal"]:
        eeg_data = _load_an_edf(
            edf_bids,
            channels=load_params["channels"],
            starts=df["onset"],
            ends=df["offset"],
        )

        if load_params.get("apply_bandpass_filter", None) not in [None, False]:
            freq_params = load_params.get("apply_bandpass_filter")
            low_freq = freq_params.get("low_freq", None)
            high_freq = freq_params.get("high_freq", None)
            bandpass_filter = BandPassFilter(
                fs=metadata_dict["SamplingFrequency"],
                low_freq=low_freq,
                high_freq=high_freq,
            )
            eeg_data = [bandpass_filter.apply(x, verbose=verbose) for x in eeg_data]

        if load_params.get("apply_notch_filter", None) not in [None, False]:
            notch_filter = NotchFilter(
                fs=metadata_dict["SamplingFrequency"],
                freqs=np.array([metadata_dict["PowerLineFrequency"]]),
            )
            eeg_data = [notch_filter.apply(x, verbose=False) for x in eeg_data]
            eeg_data = [arr for arr in eeg_data if ~np.isnan(arr).any()]

        if load_params["is_unipolar"] and load_params["apply_common-average"]:
            eeg_data = [x - np.mean(x, axis=0, keepdims=True) for x in eeg_data]

        # bipolar
        if not load_params["is_unipolar"]:
            eeg_data = [
                x[load_params["montage2channels"][0]]
                - x[load_params["montage2channels"][1]]
                for x in eeg_data
            ]

        df["eeg"] = eeg_data
    else:
        df["eeg"] = [[] for _ in range(len(df))]

    df = pd.DataFrame(df[["eeg", "trial_type"]])

    ## Labels
    session_df = session_bids.get_df()
    for k, v in session_df.items():
        assert v.shape == (1,)
        df[k] = v[0]
    del df[
        "session_id"
    ]  # session is coded in edf entities and will be embeded in the below.

    entities = edf_bids.get_entities()
    df["subject"] = entities["subject"]
    df["run"] = entities["run"]
    df["session"] = entities["session"]
    df["task"] = entities["task"]

    return df





def _load_an_edf(bf_edf, channels, starts=[0], ends=[None]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mne_raw = read_raw_edf(bf_edf, verbose=False)

    sfreq = mne_raw.info["sfreq"]

    ################################################################################
    # Kochi, override sfreq if the recording date is old.
    ss, ee = re.search("/sub-EEG\w*/", bf_edf.path).span()
    sub_ID = bf_edf.path[ss:ee][5:-1]

    if "EEGK" in sub_ID:  # Kochi
        sub_ID_wo_EEGK = sub_ID[4:]
        kochi_rec_dates = mngs.io.load("./data/secret/Kochi_Recording_Dates.xlsx", show=False)
        date = kochi_rec_dates[kochi_rec_dates["ID"] == int(sub_ID_wo_EEGK)][
            "Rec. Date 1"
        ].iloc[0]
        is_before = (date - datetime.datetime(2021, 9, 27)).days < 0
        # print(date, is_before)

        if is_before:
            sfreq = 400
    ################################################################################

    # print(sfreq)

    data = [
        mne_raw.get_data(
            picks=channels, start=int(start * sfreq), stop=int(end * sfreq)
        )
        for start, end in zip(starts, ends)
    ]

    # upsampling
    if sfreq != 500:

        n_samples = int(data[0].shape[-1] * 500 / sfreq)

        # dd = data[0]
        data = [resample(dd, n_samples, axis=-1) for dd in data]

    return data


def _add_age_sex_and_MMSE(data_all, BIDS_ROOT):
    demographic_data = (
        mngs.io.load("./data/secret/All_Institutions_List_2.xlsx", skiprows=1)
        .set_index("EEG_ID")
        .sort_index()
    ).rename(columns={"MMS": "MMSE"})
    demographic_data.index = [
        re.sub("EEGN", "EEGN0", sub) for sub in demographic_data.index
    ]

    for key in ["age", "sex", "MMSE"]:
        data_all[key] = list(demographic_data.loc[data_all["subject"]][key])

    sex_str2int_dict = {"F": -0.5, "M": 0.5}
    data_all["sex"] = data_all["sex"].replace(sex_str2int_dict)
    return data_all


# def _mask_age_sex_and_MMSE(data_all_df, load_params):
#     if load_params["no_sig"]:
#         print("\nsignal was masked\n")
#         data_all_df["eeg"] = [
#             np.random.randn(*arr.shape) for i_arr, arr in enumerate(data_all_df["eeg"])
#         ]

#     if load_params["no_age"]:
#         print("\nage was masked\n")
#         data_all_df["age_normed"] = np.random.randn(len(data_all_df["age"]))

#     if load_params["no_sex"]:
#         print("\nsex was masked\n")
#         data_all_df["sex_normed"] = 0.5

#     if load_params["no_MMSE"]:
#         print("\nMMSE was masked\n")
#         data_all_df["MMSE_normed"] = np.random.randn(len(data_all_df["MMSE"]))

#     return data_all_df


def _to_numeric(data_all_df):
    data_all_df["MMSE"][data_all_df["MMSE"] == "none"] = np.nan
    data_all_df["MMSE"] = data_all_df["MMSE"].astype(float)
    return data_all_df


def _rename_cols(data_all):
    # disease_level to cognitive_level; Nissei
    try:
        data_all["cognitive_level"] = data_all["disease_level"]
        del data_all["disease_level"]
    except:
        pass

    # MCI to cMCI; MCI23 to dMCI
    data_all["cognitive_level"] = data_all["cognitive_level"].replace(
        {"MCI": "cMCI", "MCI23": "dMCI"}
    )

    # HV to Normal on cognitive_level; for Kochi
    data_all["cognitive_level"] = data_all["cognitive_level"].replace({"HV": "Normal"})
    return data_all


def _drop_data(data):
    """
    data_all = data
    data_uq = data_all[data_all.columns[1:]].drop_duplicates()
    data = data_uq
    """

    # disease_type
    drop_indi = ~mngs.general.search(
        ["HV", "AD", "DLB", "NPH"], data["disease_type"], as_bool=True
    )[0]

    # cognitive_level
    drop_indi += ~mngs.general.search(
        ["Normal", "cMCI", "dMCI", "Dementia"], data["cognitive_level"], as_bool=True
    )[0]

    # NPH with Normal label # 3
    drop_indi += (
        mngs.general.search(["AD", "DLB", "NPH"], data["disease_type"], as_bool=True)[0]
        * mngs.general.search(["Normal"], data["cognitive_level"], as_bool=True)[0]
    )  # 3 + 0 + 0

    # age under 50
    drop_indi += data["age"] < 40  # 14 + 1 + 1

    # nan
    drop_indi += data["age"].isna()  # 1 + 0 + 0
    drop_indi += data["sex"].isna()  # 0 + 0 + 0
    drop_indi += data["MMSE"] == "none"  # 0 + 0 + 0
    drop_indi += data["MMSE"].isna()  # 4 + 0 + 0

    return data[~drop_indi]


if __name__ == "__main__":
    import mngs

    # Parameters for loading data
    load_params = mngs.io.load("./config/load_params.yaml")
    load_params["BIDS_root"] = "./data/BIDS_Kochi"
    load_params["from_pkl"] = False

    # Loads a BIDS dataset as a DataFrame
    data_all = load_BIDS(load_params, drop_unrelated=True)
    """
    # print(data_all.columns)
    Index(['eeg', 'trial_type', 'disease_type', 'pet', 'subject', 'run', 'session',
           'task', 'age', 'sex', 'MMSE', 'cognitive_level'],
          dtype='object')
    """    
