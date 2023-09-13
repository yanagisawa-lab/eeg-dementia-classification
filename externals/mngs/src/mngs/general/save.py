#!/usr/bin/env python3

import csv

import pandas as pd
import mngs
import numpy as np

import warnings


if "general" in __file__:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            '\n"mngs.general.save" will be removed. '
            'Please use "mngs.io.save" instead.',
            PendingDeprecationWarning,
        )


def save(obj, sfname_or_spath, makedirs=True, show=True, **kwargs):
    """
    Example
      save(arr, 'data.npy')
      save(df, 'df.csv')
      save(serializable, 'serializable.pkl')
    """
    import inspect
    import os
    import pickle

    import h5py
    import numpy as np
    import pandas as pd
    import torch
    import yaml

    spath, sfname = None, None

    if "/" in sfname_or_spath:
        spath = sfname_or_spath
    else:
        sfname = sfname_or_spath

    if (spath is None) and (sfname is not None):
        ## for ipython
        __file__ = inspect.stack()[1].filename
        if "ipython" in __file__:
            __file__ = f'/tmp/fake-{os.getenv("USER")}.py'

        ## spath
        fpath = __file__
        fdir, fname, _ = mngs.general.split_fpath(fpath)
        sdir = fdir + fname + "/"
        spath = sdir + sfname
        # spath = mk_spath(sfname, makedirs=True)

    ## Make directory
    if makedirs:
        sdir = os.path.dirname(spath)
        os.makedirs(sdir, exist_ok=True)

    ## Saves
    try:
        ## copy files
        is_copying_files = (isinstance(obj, str) or is_listed_X(obj, str)) and (
            isinstance(spath, str) or is_listed_X(spath, str)
        )
        if is_copying_files:
            mngs.general.copy_files(obj, spath)

        # csv
        elif spath.endswith(".csv"):
            if isinstance(obj, pd.DataFrame):  # DataFrame
                obj.to_csv(spath)
            if is_listed_X(obj, [int, float]):  # listed scalars
                save_listed_scalars_as_csv(
                    obj,
                    spath,
                    **kwargs,
                )
            if is_listed_X(obj, pd.DataFrame):  # listed DataFrame
                # save_listed_dfs_as_csv(obj, spath, indi_suffix=None, overwrite=False)
                save_listed_dfs_as_csv(obj, spath, **kwargs)
        # numpy
        elif spath.endswith(".npy"):
            np.save(spath, obj)
        # pkl
        elif spath.endswith(".pkl"):
            with open(spath, "wb") as s:  # 'w'
                pickle.dump(obj, s)
        # joblib
        elif spath.endswith(".joblib"):
            with open(spath, "wb") as s:  # 'w'
                joblib.dump(obj, s, compress=3)
        # png
        elif spath.endswith(".png"):
            # plotly
            if str(type(obj)) == "<class 'plotly.graph_objs._figure.Figure'>":
                obj.write_image(file=spath, format="png")
            # matplotlib
            else:
                try:
                    obj.savefig(spath)
                except:
                    obj.figure.savefig(spath)
            del obj
        # tiff
        elif spath.endswith(".tiff") or spath.endswith(".tif"):
            obj.savefig(
                spath, dpi=300, format="tiff"
            )  # obj is matplotlib.pyplot object
            del obj
        # mp4
        elif spath.endswith(".mp4"):
            mk_mp4(obj, spath)  # obj is matplotlib.pyplot.figure object
            del obj
        # yaml
        elif spath.endswith(".yaml"):
            with open(spath, "w") as f:
                yaml.dump(obj, f)
        # hdf5
        elif spath.endswith(".hdf5"):
            name_list, obj_list = []
            for k, v in obj.items():
                name_list.append(k)
                obj_list.append(v)
            with h5py.File(spath, "w") as hf:
                for (name, obj) in zip(name_list, obj_list):
                    hf.create_dataset(name, data=obj)
        # pth
        elif spath.endswith(".pth"):
            torch.save(obj, spath)

        # catboost model
        elif spath.endswith(".cbm"):
            obj.save_model(spath)

        else:
            raise ValueError("obj was not saved.")

    except Exception as e:
        print(e)

    else:
        if show and not is_copying_files:
            # if show:
            print(f"\nSaved to: {spath}\n")


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


def is_listed_X(obj, types):
    """
    Example:
        obj = [3, 2, 1, 5]
        is_listed_X(obj,
    """
    import numpy as np

    try:
        conditions = []
        condition_list = isinstance(obj, list)

        if not (isinstance(types, list) or isinstance(types, tuple)):
            types = [types]

        _conditions_susp = []
        for typ in types:
            _conditions_susp.append(
                (np.array([isinstance(o, typ) for o in obj]) == True).all()
            )

        condition_susp = np.any(_conditions_susp)

        is_listed_X = np.all([condition_list, condition_susp])
        return is_listed_X

    except:
        return False


def save_listed_scalars_as_csv(
    listed_scalars,
    spath_csv,
    column_name="_",
    indi_suffix=None,
    round=3,
    overwrite=False,
    show=False,
):
    """Puts to df and save it as csv"""
    import numpy as np

    if overwrite == True:
        mv_to_tmp(spath_csv, L=2)
    indi_suffix = np.arange(len(listed_scalars)) if indi_suffix is None else indi_suffix
    df = pd.DataFrame(
        {"{}".format(column_name): listed_scalars}, index=indi_suffix
    ).round(round)
    df.to_csv(spath_csv)
    if show:
        print("\nSaved to: {}\n".format(spath_csv))


def save_listed_dfs_as_csv(
    listed_dfs,
    spath_csv,
    indi_suffix=None,
    overwrite=False,
    show=False,
):
    """listed_dfs:
        [df1, df2, df3, ..., dfN]. They will be written vertically in the order.

    spath_csv:
        /hoge/fuga/foo.csv

    indi_suffix:
        At the left top cell on the output csv file, '{}'.format(indi_suffix[i])
        will be added, where i is the index of the df.On the other hand,
        when indi_suffix=None is passed, only '{}'.format(i) will be added.
    """
    import numpy as np

    if overwrite == True:
        mv_to_tmp(spath_csv, L=2)

    indi_suffix = np.arange(len(listed_dfs)) if indi_suffix is None else indi_suffix
    for i, df in enumerate(listed_dfs):
        with open(spath_csv, mode="a") as f:
            f_writer = csv.writer(f)
            i_suffix = indi_suffix[i]
            f_writer.writerow(["{}".format(indi_suffix[i])])
        df.to_csv(spath_csv, mode="a", index=True, header=True)
        with open(spath_csv, mode="a") as f:
            f_writer = csv.writer(f)
            f_writer.writerow([""])
    if show:
        print("Saved to: {}".format(spath_csv))


def mk_mp4(fig, spath_mp4):
    from matplotlib import animation

    axes = fig.get_axes()

    def init():
        return (fig,)

    def animate(i):
        for ax in axes:
            ax.view_init(elev=10.0, azim=i)
        return (fig,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=360, interval=20, blit=True
    )

    writermp4 = animation.FFMpegWriter(fps=60, extra_args=["-vcodec", "libx264"])
    anim.save(spath_mp4, writer=writermp4)
    print("\nSaving to: {}\n".format(spath_mp4))


def mv_to_tmp(fpath, L=2):
    import os
    from shutil import move

    try:
        tgt_fname = connect_strs_with_hyphens(fpath.split("/")[-L:])
        tgt_fpath = "/tmp/{}".format(tgt_fname)
        move(fpath, tgt_fpath)
        print("Moved to: {}".format(tgt_fpath))
    except:
        pass


def save_optuna_study_as_csv_and_pngs(study, sdir):
    import optuna

    ## Trials DataFrame
    trials_df = study.trials_dataframe()

    ## Figures
    hparams_keys = list(study.best_params.keys())
    slice_plot = optuna.visualization.plot_slice(study, params=hparams_keys)
    contour_plot = optuna.visualization.plot_contour(study, params=hparams_keys)
    optim_hist_plot = optuna.visualization.plot_optimization_history(study)
    parallel_coord_plot = optuna.visualization.plot_parallel_coordinate(
        study, params=hparams_keys
    )
    hparam_importances_plot = optuna.visualization.plot_param_importances(study)
    figs_dict = dict(
        slice_plot=slice_plot,
        contour_plot=contour_plot,
        optim_hist_plot=optim_hist_plot,
        parallel_coord_plot=parallel_coord_plot,
        hparam_importances_plot=hparam_importances_plot,
    )

    ## Saves
    save(trials_df, sdir + "trials_df.csv")

    for figname, fig in figs_dict.items():
        save(fig, sdir + f"{figname}.png")
