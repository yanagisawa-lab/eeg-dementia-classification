#!/usr/bin/env python3
# Time-stamp: "2021-11-27 18:45:23 (ylab)"

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def mk_patches(colors, labels):
    """
    colors = ["red", "blue"]
    labels = ["label_1", "label_2"]
    ax.legend(handles=mngs.plt.mk_patches(colors, labels))
    """

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    return patches
