#!/usr/bin/env python3

import mngs
import warnings


if "general" in __file__:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            '\n"mngs.general.load" will be removed. '
            'Please use "mngs.io.load" instead.',
            PendingDeprecationWarning,
        )


def load(lpath, show=False, **kwargs):
    import pickle

    import h5py
    import numpy as np
    import pandas as pd
    import torch
    import yaml

    # csv
    if lpath.endswith(".csv"):
        obj = pd.read_csv(lpath, **kwargs)
    # numpy
    if lpath.endswith(".npy"):
        obj = np.load(lpath)
    # pkl
    if lpath.endswith(".pkl"):
        with open(lpath, "rb") as l:
            obj = pickle.load(l)
    # joblib
    if lpath.endswith(".joblib"):
        with open(lpath, "rb") as l:
            obj = joblib.load(l)
    # hdf5
    if lpath.endswith(".hdf5"):
        obj = {}
        with h5py.File(fpath, "r") as hf:
            for name in name_list:
                obj_tmp = hf[name][:]
                obj[name] = obj_tmp
    # png
    if lpath.endswith(".png"):
        pass
    # tiff
    if lpath.endswith(".tiff") or lpath.endswith(".tif"):
        pass
    # yaml
    if lpath.endswith(".yaml"):
        obj = {}
        with open(lpath) as f:
            obj_tmp = yaml.safe_load(f)
            obj.update(obj_tmp)
    # txt
    if lpath.endswith(".txt"):
        f = open(lpath, "r")
        obj = [l.strip("\n\r") for l in f]
        f.close()
    # pth
    if lpath.endswith(".pth"):
        # return model.load_state_dict(torch.load(lpath))
        obj = torch.load(lpath)

    if lpath.endswith(".mat"):
        import pymatreader

        obj = pymatreader.read_mat(lpath)
    # xml
    if lpath.endswith("xml"):
        from ._xml2dict import xml2dict

        obj = xml2dict(lpath)

    # catboost model
    if lpath.endswith(".cbm"):
        obj = obj.load_model(lpath)
        
    # if mngs.general.is_defined_local("obj"):
    if "obj" in locals():
        print("\nLoaded from: {}\n".format(lpath))
        return obj
    else:
        return None


def check_encoding(file_path):
    from chardet.universaldetector import UniversalDetector

    detector = UniversalDetector()
    with open(file_path, mode="rb") as f:
        for binary in f:
            detector.feed(binary)
            if detector.done:
                break
    detector.close()
    enc = detector.result["encoding"]
    return enc


def get_data_path_from_a_package(package_str, resource):
    import importlib
    import os
    import sys

    spec = importlib.util.find_spec(package_str)
    data_dir = os.path.join(spec.origin.split("src")[0], "data")
    resource_path = os.path.join(data_dir, resource)
    return resource_path

def load_yaml_as_an_optuna_dict(fpath_yaml, trial):
    _d = load(fpath_yaml)    
    
    for k, v in _d.items():
        
        dist = v["distribution"]

        if dist == "categorical":
            _d[k] = trial.suggest_categorical(k, v["values"])

        elif dist == "uniform":
            _d[k] = trial.suggest_int(k, float(v["min"]), float(v["max"]))
            
        elif dist == "loguniform":
            _d[k] = trial.suggest_loguniform(k, float(v["min"]), float(v["max"]))

        elif dist == "intloguniform":
            _d[k] = trial.suggest_int(k, float(v["min"]), float(v["max"]), log=True)
            
    return _d

def load_study_rdb(study_name, rdb_raw_bytes_url):
    """
    study = load_study_rdb(
        study_name="YOUR_STUDY_NAME",
        rdb_raw_bytes_url="sqlite:///*.db"
    )
    """
    import optuna
    # rdb_raw_bytes_url = "sqlite:////tmp/fake/ywatanabe/_MicroNN_WindowSize-1.0-sec_MaxEpochs_100_2021-1216-1844/optuna_study_test_file#0.db"
    storage = optuna.storages.RDBStorage(url=rdb_raw_bytes_url)
    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"\nLoaded: {rdb_raw_bytes_url}\n")
    return study
    
    
