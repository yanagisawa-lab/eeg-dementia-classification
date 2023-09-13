#!/usr/bin/env python3

import re
from collections import defaultdict
from pprint import pprint

import matplotlib
import mngs
import pandas as pd
import numpy as np
import warnings


class LearningCurveLogger(object):
    def __init__(
        self,
    ):
        self.logged_dict = defaultdict(dict)

        warnings.warn(
            '\n"gt_label" will be removed in the feature. Please use "true_class" instead.\n',
            DeprecationWarning,
        )

    def __call__(self, dict_to_log, step):
        """
        dict_to_log | str
            Example:
                dict_to_log = {
                    "loss_plot": float(loss),
                    "balanced_ACC_plot": float(bACC),
                    "pred_proba": pred_proba.detach().cpu().numpy(),
                    "true_class": T.cpu().numpy(),
                    "i_fold": i_fold,
                    "i_epoch": i_epoch,
                    "i_global": i_global,
                }

        step | str
            "Training", "Validation", or "Test"
        """
        ########################################
        ## delete here in the future
        ## rename "gt_label" to "true_class"
        if "gt_label" in dict_to_log.keys():
            dict_to_log["true_class"] = dict_to_log.pop("gt_label")
            # del dict_to_log["gt_label"]
        ########################################

        assert step in ["Training", "Validation", "Test"]

        ## Initialize self.logged_dict
        for k_to_log in dict_to_log.keys():
            try:
                self.logged_dict[step][k_to_log].append(dict_to_log[k_to_log])
            except:
                self.logged_dict[step].update({k_to_log: []})
                self.logged_dict[step][k_to_log].append(dict_to_log[k_to_log])

    @property
    def dfs(self):
        return self._to_dfs_pivot(
            self.logged_dict,
            pivot_column=None,
        )

    def get_x_of_i_epoch(self, x, step, i_epoch):
        """
        lc_logger.get_x_of_i_epoch("true_label_diag", "Validation", 3)
        """
        assert step in ['Training', "Validation", "Test"]
        indi = np.array(self.logged_dict[step]['i_epoch']) == i_epoch
        x_all_arr = np.array(self.logged_dict[step][x])
        assert len(indi) == len(x_all_arr)
        return x_all_arr[indi]


    def plot_learning_curves(
        self,
        plt,
        plt_config_dict=None,
        title=None,
        max_n_ticks=4,
        linewidth=1,
        scattersize=50,
    ):
        """
        Plots learning curves from self.logged_dict
        """
        if plt_config_dict is not None:
            mngs.plt.configure_mpl(plt, **plt_config_dict)

        self.dfs_pivot_i_global = self._to_dfs_pivot(
            self.logged_dict, pivot_column="i_global"
        )

        ########################################
        ## Parameters
        ########################################
        COLOR_DICT = {
            "Training": "blue",
            "Validation": "green",
            "Test": "red",
        }

        keys_to_plot = self._find_keys_to_plot(self.logged_dict)

        ########################################
        ## Plots
        ########################################
        fig, axes = plt.subplots(len(keys_to_plot), 1, sharex=True, sharey=False)
        axes[-1].set_xlabel("Iteration#")
        fig.text(0.5, 0.95, title, ha="center")

        for i_plt, plt_k in enumerate(keys_to_plot):
            ax = axes[i_plt]
            ax.set_ylabel(self._rename_if_key_to_plot(plt_k))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_n_ticks))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_n_ticks))

            if re.search("[aA][cC][cC]", plt_k):  # acc, ylim, yticks
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 0.5, 1.0])

            for step_k in self.dfs_pivot_i_global.keys():

                if step_k == "Training":  # line
                    ax.plot(
                        self.dfs_pivot_i_global[step_k].index,
                        self.dfs_pivot_i_global[step_k][plt_k],
                        label=step_k,
                        color=mngs.plt.colors.to_RGBA(COLOR_DICT[step_k], alpha=0.9),
                        linewidth=linewidth,
                    )
                    ax.legend()

                    ########################################
                    ## Epoch starts points; just in "Training" not to b duplicated
                    ########################################
                    epoch_starts = abs(
                        self.dfs_pivot_i_global[step_k]["i_epoch"]
                        - self.dfs_pivot_i_global[step_k]["i_epoch"].shift(-1)
                    )
                    indi_global_epoch_starts = [0] + list(
                        epoch_starts[epoch_starts == 1].index
                    )

                    for i_epoch, i_global_epoch_start in enumerate(
                        indi_global_epoch_starts
                    ):
                        ax.axvline(
                            x=i_global_epoch_start,
                            ymin=-1e4,  # ax.get_ylim()[0],
                            ymax=1e4,  # ax.get_ylim()[1],
                            linestyle="--",
                            color=mngs.plt.colors.to_RGBA("gray", alpha=0.5),
                        )
                        # ax.text(
                        #     i_global_epoch_start,
                        #     ax.get_ylim()[1] * 1.0,
                        #     f"epoch#{i_epoch}",
                        # )
                    ########################################

                if (step_k == "Validation") or (step_k == "Test"):  # scatter
                    ax.scatter(
                        self.dfs_pivot_i_global[step_k].index,
                        self.dfs_pivot_i_global[step_k][plt_k],
                        label=step_k,
                        color=mngs.plt.colors.to_RGBA(COLOR_DICT[step_k], alpha=0.9),
                        s=scattersize,
                        alpha=0.9,
                    )
                    ax.legend()

        return fig

    def print(self, step):
        df_pivot_i_epoch = self._to_dfs_pivot(self.logged_dict, pivot_column="i_epoch")
        df_pivot_i_epoch_step = df_pivot_i_epoch[step]
        df_pivot_i_epoch_step.columns = self._rename_if_key_to_plot(
            df_pivot_i_epoch_step.columns
        )
        print("\n----------------------------------------\n")
        print(f"\n{step}: (mean of batches)\n")
        pprint(df_pivot_i_epoch_step)
        print("\n----------------------------------------\n")


    @staticmethod
    def _find_keys_to_plot(logged_dict):
        _steps_str = list(logged_dict.keys())
        _, keys_to_plot = mngs.general.search(
            "_plot",
            list(logged_dict[_steps_str[0]].keys()),
        )
        return keys_to_plot

    @staticmethod
    def _rename_if_key_to_plot(keys):
        def _rename_key_to_plot(key_to_plot):
            renamed = key_to_plot[:-5]
            renamed = renamed.replace("_", " ")
            capitalized = []
            for s in renamed.split(" "):
                if not re.search("^[A-Z]", s):
                    capitalized.append(s.capitalize())
                else:
                    capitalized.append(s)
            renamed = mngs.general.connect_strs(capitalized, filler=" ")
            return renamed

        if isinstance(keys, str):
            keys = [keys]

        out = []
        for key in keys:
            if key[-5:] == "_plot":
                out.append(_rename_key_to_plot(key))
            else:
                out.append(key)

        if len(out) == 1:
            out = out[0]

        return out

    # @staticmethod
    # def _to_dfs_pivot_i_global(logged_dict):
    #     dfs_pivot_i_global = {}
    #     for step in logged_dict.keys():
    #         df_step = mngs.general.pandas.force_dataframe(logged_dict[step])
    #         df_step_pvt_on_i_global = df_step.pivot_table(
    #             columns="i_global", aggfunc="mean"
    #         ).T
    #         dfs_pivot_i_global[step] = df_step_pvt_on_i_global

    #     return dfs_pivot_i_global

    @staticmethod
    def _to_dfs_pivot(logged_dict, pivot_column=None):
        dfs_pivot = {}
        for step in logged_dict.keys():
            df_step = mngs.general.pandas.force_dataframe(logged_dict[step])
            if pivot_column is not None:
                dfs_pivot[step] = df_step.pivot_table(
                    columns=pivot_column, aggfunc="mean"
                ).T
            else:
                dfs_pivot[step] = df_step

        return dfs_pivot


if __name__ == "__main__":
    import warnings

    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from sklearn.metrics import balanced_accuracy_score
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.dataset import Subset
    from torchvision import datasets

    import sys

    ################################################################################
    ## Sets tee
    ################################################################################
    sdir = mngs.io.path.mk_spath("")  # "/tmp/sdir/"
    sys.stdout, sys.stderr = mngs.general.tee(sys, sdir)

    ################################################################################
    ## NN
    ################################################################################
    class Perceptron(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(28 * 28, 50)
            self.l2 = nn.Linear(50, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.l1(x)
            x = self.l2(x)
            return x

    ################################################################################
    ## Prepaires demo data
    ################################################################################
    ## Downloads
    _ds_tra_val = datasets.MNIST("/tmp/mnist", train=True, download=True)
    _ds_tes = datasets.MNIST("/tmp/mnist", train=False, download=True)

    ## Training-Validation splitting
    n_samples = len(_ds_tra_val)  # n_samples is 60000
    train_size = int(n_samples * 0.8)  # train_size is 48000

    subset1_indices = list(range(0, train_size))  # [0,1,.....47999]
    subset2_indices = list(range(train_size, n_samples))  # [48000,48001,.....59999]

    _ds_tra = Subset(_ds_tra_val, subset1_indices)
    _ds_val = Subset(_ds_tra_val, subset2_indices)

    ## to tensors
    ds_tra = TensorDataset(
        _ds_tra.dataset.data.to(torch.float32),
        _ds_tra.dataset.targets,
    )
    ds_val = TensorDataset(
        _ds_val.dataset.data.to(torch.float32),
        _ds_val.dataset.targets,
    )
    ds_tes = TensorDataset(
        _ds_tes.data.to(torch.float32),
        _ds_tes.targets,
    )

    ## to dataloaders
    batch_size = 64
    dl_tra = DataLoader(
        dataset=ds_tra,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    dl_val = DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    dl_tes = DataLoader(
        dataset=ds_tes,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    ################################################################################
    ## Preparation
    ################################################################################
    model = Perceptron()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    softmax = nn.Softmax(dim=-1)

    ################################################################################
    ## Main
    ################################################################################
    lc_logger = LearningCurveLogger()
    i_global = 0

    n_classes = len(dl_tra.dataset.tensors[1].unique())
    i_fold = 0
    max_epochs = 3

    for i_epoch in range(max_epochs):
        step = "Validation"
        for i_batch, batch in enumerate(dl_val):

            X, T = batch
            logits = model(X)
            pred_proba = softmax(logits)
            pred_class = pred_proba.argmax(dim=-1)
            loss = loss_func(logits, T)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bACC = balanced_accuracy_score(T, pred_class)

            dict_to_log = {
                "loss_plot": float(loss),
                "balanced_ACC_plot": float(bACC),
                "pred_proba": pred_proba.detach().cpu().numpy(),
                "gt_label": T.cpu().numpy(),
                # "true_class": T.cpu().numpy(),
                "i_fold": i_fold,
                "i_epoch": i_epoch,
                "i_global": i_global,
            }
            lc_logger(dict_to_log, step)

        lc_logger.print(step)

        step = "Training"
        for i_batch, batch in enumerate(dl_tra):
            optimizer.zero_grad()

            X, T = batch
            logits = model(X)
            pred_proba = softmax(logits)
            pred_class = pred_proba.argmax(dim=-1)
            loss = loss_func(logits, T)

            loss.backward()
            optimizer.step()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bACC = balanced_accuracy_score(T, pred_class)

            dict_to_log = {
                "loss_plot": float(loss),
                "balanced_ACC_plot": float(bACC),
                "pred_proba": pred_proba.detach().cpu().numpy(),
                "gt_label": T.cpu().numpy(),
                # "true_class": T.cpu().numpy(),
                "i_fold": i_fold,
                "i_epoch": i_epoch,
                "i_global": i_global,
            }
            lc_logger(dict_to_log, step)

            i_global += 1

        lc_logger.print(step)

    step = "Test"
    for i_batch, batch in enumerate(dl_tes):

        X, T = batch
        logits = model(X)
        pred_proba = softmax(logits)
        pred_class = pred_proba.argmax(dim=-1)
        loss = loss_func(logits, T)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bACC = balanced_accuracy_score(T, pred_class)

        dict_to_log = {
            "loss_plot": float(loss),
            "balanced_ACC_plot": float(bACC),
            "pred_proba": pred_proba.detach().cpu().numpy(),
            # "gt_label": T.cpu().numpy(),
            "true_class": T.cpu().numpy(),
            "i_fold": i_fold,
            "i_epoch": i_epoch,
            "i_global": i_global,
        }
        lc_logger(dict_to_log, step)

    lc_logger.print(step)

    plt_config_dict = dict(
        # figsize=(8.7, 10),
        figscale=2.5,
        labelsize=16,
        fontsize=12,
        legendfontsize=12,
        tick_size=0.8,
        tick_width=0.2,
    )

    fig = lc_logger.plot_learning_curves(
        plt,
        plt_config_dict=plt_config_dict,
        title=f"fold#{i_fold}",
        linewidth=1,
        scattersize=50,
    )
    fig.show()
    # mngs.general.save(fig, sdir + f"fold#{i_fold}.png")
