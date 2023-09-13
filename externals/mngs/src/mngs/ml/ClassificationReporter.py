#!/usr/bin/env python3

import os
import random
import sys
from collections import defaultdict
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)


class ClassificationReporter(object):
    """Saves the following metrics under sdir.
       - Balanced Accuracy
       - MCC
       - Confusion Matrix
       - Classification Report
       - ROC AUC score / curve
       - PRE-REC AUC score / curve

    Example is described in this file.
    """

    def __init__(self, sdir):
        self.sdir = sdir
        self.folds_dict = defaultdict(list)
        mngs.general.fix_seeds(os=os, random=random, np=np, torch=torch, show=False)

    def add(
        self,
        obj_name,
        obj,
    ):
        """
        ## fig
        fig, ax = plt.subplots()
        ax.plot(np.random.rand(10))
        reporter.add("manu_figs", fig)

        ## DataFrame
        df = pd.DataFrame(np.random.rand(5, 3))
        reporter.add("manu_dfs", df)

        ## scalar
        scalar = random.random()
        reporter.add("manu_scalers", scalar)
        """
        assert isinstance(obj_name, str)
        self.folds_dict[obj_name].append(obj)

    def calc_metrics(
        self,
        true_class,
        pred_class,
        pred_proba,
        labels=None,
        i_fold=None,
        show=True,
        auc_plt_config=dict(
            figsize=(7, 7),
            labelsize=8,
            fontsize=7,
            legendfontsize=6,
            tick_size=0.8,
            tick_width=0.2,
        ),
    ):
        """Calculates ACC, Confusion Matrix, Classification Report, and ROC-AUC score."""
        self.labels = labels

        true_class = mngs.general.torch_to_arr(true_class).astype(int).reshape(-1)
        pred_class = (
            mngs.general.torch_to_arr(pred_class).astype(np.float64).reshape(-1)
        )
        pred_proba = mngs.general.torch_to_arr(pred_proba).astype(np.float64)

        ####################
        ## Scalar metrices
        ####################
        balanced_acc = balanced_accuracy_score(true_class, pred_class)
        mcc = float(matthews_corrcoef(true_class, pred_class))

        if show:
            print(f"\nBalanced ACC in fold#{i_fold} was {balanced_acc:.3f}\n")
            print(f"\nMCC in fold#{i_fold} was {mcc:.3f}\n")

        ####################
        ## Confusion Matrix
        ####################
        conf_mat = pd.DataFrame(
            data=confusion_matrix(true_class, pred_class), columns=labels
        ).set_index(pd.Series(list(labels)))
        if show:
            print(f"\nConfusion Matrix in fold#{i_fold}: \n")
            pprint(conf_mat)
            print()

        ####################
        ## Classification Report
        ####################
        clf_report = pd.DataFrame(
            classification_report(
                true_class,
                pred_class,
                target_names=labels,
                output_dict=True,
            )
        )
        ## ACC to bACC
        clf_report["accuracy"] = balanced_acc
        clf_report = clf_report.rename(columns={"accuracy": "balanced accuracy"})
        clf_report = clf_report.round(3)
        ## Renames 'support' to 'sample size'
        clf_report["index"] = clf_report.index
        clf_report.loc["support", "index"] = "sample size"
        clf_report.set_index("index", drop=True, inplace=True)
        clf_report.index.name = None
        if show:
            print(f"\nClassification Report for fold#{i_fold}:\n")
            pprint(clf_report)
            print()

        ####################
        ## ROC-AUC score
        ####################
        mngs.plt.configure_mpl(
            plt,
            **auc_plt_config,
        )

        fig_roc, metrics_roc_auc_dict = mngs.ml.plt.roc_auc(
            plt,
            true_class,
            pred_proba,
            labels,
        )
        plt.close()

        ####################
        ## PRE-REC AUC score
        ####################
        fig_pre_rec, metrics_pre_rec_auc_dict = mngs.ml.plt.pre_rec_auc(
            plt, true_class, pred_proba, labels
        )
        plt.close()

        ####################
        ## to the buffer
        ####################
        self.folds_dict["mcc"].append(mcc)
        self.folds_dict["balanced_acc"].append(balanced_acc)
        self.folds_dict["conf_mat"].append(conf_mat)
        self.folds_dict["clf_report"].append(clf_report)
        self.folds_dict["roc_auc_micro"].append(
            metrics_roc_auc_dict["roc_auc"]["micro"]
        )
        self.folds_dict["roc_auc_macro"].append(
            metrics_roc_auc_dict["roc_auc"]["macro"]
        )
        self.folds_dict["roc_auc_fig"].append(fig_roc)

        self.folds_dict["pre_rec_auc_micro"].append(
            metrics_pre_rec_auc_dict["pre_rec_auc"]["micro"]
        )
        self.folds_dict["pre_rec_auc_macro"].append(
            metrics_pre_rec_auc_dict["pre_rec_auc"]["macro"]
        )
        self.folds_dict["pre_rec_auc_fig"].append(fig_pre_rec)

    def summarize(
        self,
        n_round=3,
        show=False,
    ):
        """
        1) Take mean and std of scalars/pd.Dataframes for folds.
        2) Replace self.folds_dict with the summarized DataFrames.
        """
        _n_folds = [len(self.folds_dict[k]) for k in self.folds_dict.keys()]
        assert len(np.unique(_n_folds)) == 1
        self.n_folds = n_folds = _n_folds[0]
        self.cv_index = [f"{n_folds}-folds_CV_mean", f"{n_folds}-fold_CV_std"] + [
            f"fold#{i_fold}" for i_fold in range(n_folds)
        ]

        for k in self.folds_dict.keys():
            ## listed scalars
            if mngs.general.is_listed_X(self.folds_dict[k], [float, int]):
                mm = np.mean(self.folds_dict[k])
                ss = np.std(self.folds_dict[k], ddof=1)

                sr = pd.DataFrame(
                    data=[mm, ss] + self.folds_dict[k], index=self.cv_index, columns=[k]
                )
                self.folds_dict[k] = sr.round(n_round)

                if show:
                    print("\n----------------------------------------\n")
                    print(f"\n{self.folds_dict[k].iloc[:2]}\n")
                    print("\n----------------------------------------\n")

            ## listed pd.DataFrames
            elif mngs.general.is_listed_X(self.folds_dict[k], pd.DataFrame):
                zero_df_for_mm = 0 * self.folds_dict[k][0].copy()
                zero_df_for_ss = 0 * self.folds_dict[k][0].copy()

                mm = (zero_df_for_mm + np.stack(self.folds_dict[k]).mean(axis=0)).round(
                    n_round
                )
                ss = (
                    zero_df_for_ss + np.stack(self.folds_dict[k]).std(axis=0, ddof=1)
                ).round(n_round)

                self.folds_dict[k] = [mm, ss] + [
                    df_fold.round(n_round) for df_fold in self.folds_dict[k]
                ]

                if show:
                    print(
                        "\n----------------------------------------\n"
                        f"\n{k}\n"
                        f"\n{n_folds}-fold-CV mean:\n"
                    )
                    pprint(self.folds_dict[k][0])
                    print(f"\n\n{n_folds}-fold-CV std.:\n")
                    pprint(self.folds_dict[k][1])
                    print("\n\n----------------------------------------\n")

            ## listed figures
            elif mngs.general.is_listed_X(self.folds_dict[k], matplotlib.figure.Figure):
                pass

            else:
                print(f"{k} was not summarized")
                print(type(self.folds_dict[k][0]))

    def save(
        self,
        files_to_reproduce=None,
        meta_dict=None,
    ):
        """
        1) Saves the content of self.folds_dict.
        2) Plots the colormap of confusion matrices and saves them.
        3) Saves passed meta_dict under self.sdir

        Example:
            meta_df_1 = pd.DataFrame(data=np.random.rand(3,3))
            meta_dict_1 = {"a": 0}
            meta_dict_2 = {"b": 0}
            meta_dict = {"meta_1.csv": meta_df_1,
                         "meta_1.yaml": meta_dict_1,
                         "meta_2.yaml": meta_dict_1,
            }

        """
        if meta_dict is not None:
            for k, v in meta_dict.items():
                mngs.io.save(v, self.sdir + k)

        for k in self.folds_dict.keys():

            ## pd.Series / pd.DataFrame
            if isinstance(self.folds_dict[k], pd.Series) or isinstance(
                self.folds_dict[k], pd.DataFrame
            ):
                mngs.io.save(self.folds_dict[k], self.sdir + f"{k}.csv")

            ## listed pd.DataFrame
            elif mngs.general.is_listed_X(self.folds_dict[k], pd.DataFrame):
                mngs.io.save(
                    self.folds_dict[k],
                    self.sdir + f"{k}.csv",
                    indi_suffix=self.cv_index,
                )

            ## listed figures
            elif mngs.general.is_listed_X(self.folds_dict[k], matplotlib.figure.Figure):
                for i_fold, fig in enumerate(self.folds_dict[k]):
                    mngs.io.save(
                        self.folds_dict[k][i_fold], self.sdir + f"{k}/fold#{i_fold}.png"
                    )

            else:
                print(f"{k} was not saved")
                print(type(self.folds_dict[k]))

        if files_to_reproduce is not None:
            if isinstance(files_to_reproduce, list):
                files_to_reproduce = [files_to_reproduce]
            for f in files_to_reproduce:
                mngs.io.save(f, self.sdir)

    def plot_and_save_conf_mats(
        self,
        plt,
        extend_ratio=1.0,
        colorbar=True,
        confmat_plt_config=None,
        sci_notation_kwargs=None,
    ):
        def _plot_conf_mat(
            plt, cm_df, title, extend_ratio=1.0, colorbar=True, sci_notation_kwargs=None
        ):
            labels = list(cm_df.columns)
            fig_conf_mat = mngs.ml.plt.confusion_matrix(
                plt,
                cm_df.T,
                labels=labels,
                title=title,
                x_extend_ratio=extend_ratio,
                y_extend_ratio=extend_ratio,                
                colorbar=colorbar,
            )

            if sci_notation_kwargs is not None:
                fig_conf_mat.axes[-1] = mngs.plt.ax_scientific_notation(
                    fig_conf_mat.axes[-1], **sci_notation_kwargs
                )
            return fig_conf_mat

        ## Configures mpl
        mngs.plt.configure_mpl(
            plt,
            **confmat_plt_config,
        )

        ########################################
        ## Prepares confmats dfs
        ########################################
        ## Drops mean and std for the folds
        try:
            conf_mats = self.folds_dict["conf_mat"][-self.n_folds :]

        except Exception as e:
            print(e)
            conf_mats = self.folds_dict["conf_mat"]

        ## Prepaires conf_mat_overall_sum
        conf_mat_zero = 0 * conf_mats[0].copy()  # get the table format
        conf_mat_overall_sum = conf_mat_zero + np.stack(conf_mats).sum(axis=0)

        ########################################
        ## Plots & Saves
        ########################################
        # each fold's conf
        for i_fold, cm in enumerate(conf_mats):
            title = f"Test fold#{i_fold}"
            fig_conf_mat_fold = _plot_conf_mat(
                plt,
                cm,
                title,
                extend_ratio=extend_ratio,
                colorbar=colorbar,
                sci_notation_kwargs=sci_notation_kwargs,
            )
            mngs.io.save(
                fig_conf_mat_fold, self.sdir + f"conf_mat_figs/fold#{i_fold}.png"
            )
            plt.close()

        ## overall_sum conf_mat
        title = f"{self.n_folds}-CV overall sum"
        fig_conf_mat_overall_sum = _plot_conf_mat(
            plt,
            conf_mat_overall_sum,
            title,
            extend_ratio=extend_ratio,
            colorbar=colorbar,
            sci_notation_kwargs=sci_notation_kwargs,
        )
        mngs.io.save(
            fig_conf_mat_overall_sum,
            self.sdir + "conf_mat_figs/k-fold_cv_overall-sum.png",
        )
        plt.close()


if __name__ == "__main__":
    import random
    import sys

    import mngs
    import numpy as np
    from catboost import CatBoostClassifier, Pool
    from sklearn.datasets import load_digits
    from sklearn.model_selection import StratifiedKFold

    ################################################################################
    ## Sets tee
    ################################################################################
    sdir = mngs.io.mk_spath("")  # "/tmp/sdir/"
    sys.stdout, sys.stderr = mngs.general.tee(sys, sdir)

    ################################################################################
    ## Fixes seeds
    ################################################################################
    mngs.general.fix_seeds(np=np)

    ## Loads
    mnist = load_digits()
    X, T = mnist.data, mnist.target
    labels = mnist.target_names.astype(str)

    ## Main
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    reporter = ClassificationReporter(sdir)
    for i_fold, (indi_tra, indi_tes) in enumerate(skf.split(X, T)):
        X_tra, T_tra = X[indi_tra], T[indi_tra]
        X_tes, T_tes = X[indi_tes], T[indi_tes]

        clf = CatBoostClassifier(verbose=False)

        clf.fit(X_tra, T_tra, verbose=False)

        ## Prediction
        pred_proba_tes = clf.predict_proba(X_tes)
        pred_cls_tes = np.argmax(pred_proba_tes, axis=1)

        ##############################
        ## Manually adds objects to reporter to save
        ##############################
        ## Figure
        fig, ax = plt.subplots()
        ax.plot(np.arange(10))
        reporter.add("manu_figs", fig)

        ## DataFrame
        df = pd.DataFrame(np.random.rand(5, 3))
        reporter.add("manu_dfs", df)

        ## Scalar
        scalar = random.random()
        reporter.add(
            "manu_scalars",
            scalar,
        )

        ########################################
        ## Metrics
        ########################################
        reporter.calc_metrics(
            T_tes,
            pred_cls_tes,
            pred_proba_tes,
            labels=labels,
            i_fold=i_fold,
        )

    reporter.summarize(show=True)

    fake_fpaths = ["fake_file_1.txt", "fake_file_2.txt"]
    for ff in fake_fpaths:
        mngs.io.touch(ff)

    files_to_reproduce = [
        mngs.general.get_this_fpath(when_ipython="/dev/null"),
        *fake_fpaths,
    ]
    reporter.save(files_to_reproduce=files_to_reproduce)

    confmat_plt_config = dict(
        figsize=(8, 8),
        # labelsize=8,
        # fontsize=6,
        # legendfontsize=6,
        figscale=2,
        tick_size=0.8,
        tick_width=0.2,
    )

    sci_notation_kwargs = dict(
        order=1,
        fformat="%1.0d",
        scilimits=(-3, 3),
        x=False,
        y=True,
    )  # "%3.1f"

    # sci_notation_kwargs = None
    reporter.plot_and_save_conf_mats(
        plt,
        extend_ratio=1.0,
        confmat_plt_config=confmat_plt_config,
        sci_notation_kwargs=sci_notation_kwargs,
    )

    ## EOF
