#!/usr/bin/env python

import mne
import numpy as np
import matplotlib.pyplot as plt
import mngs


def plot_topomap(
    data,
    CH_MONTAGES_WO_REF,
    vmin=None,
    vmax=None,
    cmap=None,
):
    # see here for colorbar
    # https://mne.discourse.group/t/mne-viz-plot-topomap-and-color-bar/3141/3

    fig, ax = plt.subplots()

    ## Gets the standard 10-20's coordinates
    ten_twenty_montage = mne.channels.make_standard_montage(
        "standard_1020",
        head_size=0.080,
    )  # default: head_size=0.095

    # print(ten_twenty_montage.get_positions().keys())
    # dict_keys(['ch_pos', 'coord_frame', 'nasion', 'lpa', 'rpa', 'hsp', 'hpi'])
    positions_dict = ten_twenty_montage.get_positions()["ch_pos"]

    CH_POSI_ARR = np.array([positions_dict[cmwor] for cmwor in CH_MONTAGES_WO_REF])

    im, cn = mne.viz.plot_topomap(
        data,
        CH_POSI_ARR[:, :2],
        vmin=vmin,
        vmax=vmax,
        names=CH_MONTAGES_WO_REF,
        show_names=True,
        extrapolate="head",
        axes=ax,
        show=False,
        cmap=cmap,
    )

    # manuallyf fiddle the position of colorbar
    ax_x_start = 0.85
    ax_x_width = 0.04
    ax_y_start = 0.1
    ax_y_height = 0.75

    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    unit_label = "aaa"
    fontsize = 12
    clb.ax.set_title(unit_label, fontsize=fontsize)

    return fig  # im, cn


if __name__ == "__main__":
    CH_MONTAGES = [
        "Fp1-A1",
        "F3-A1",
        "C3-A1",
        "P3-A1",
        "O1-A1",
        "Fp2-A2",
        "F4-A2",
        "C4-A2",
        "P4-A2",
        "O2-A2",
        "F7-A1",
        "T7-A1",
        "P7-A1",
        "F8-A2",
        "T8-A2",
        "P8-A2",
        "Fz-A1",
        "Cz-A1",
        "Pz-A1",
    ]

    N_CHS = len(CH_MONTAGES)

    # https://mne.tools/stable/generated/mne.channels.Layout.html#mne.channels.Layout

    ## Creates demo data
    data = np.random.rand(N_CHS)

    ## Gets the 19 chs' coordinates
    CH_MONTAGES_WO_REF = [ch_montage.split("-")[0] for ch_montage in CH_MONTAGES]
    # CH_POSITIONS_DICT = {ch_montage_wo_ref:positions_dict[ch_montage_wo_ref]
    #                      for ch_montage_wo_ref in CH_MONTAGES_WO_REF}

    ## Plot
    # im, cn = plot_topomap(data, CH_MONTAGES_WO_REF, vmin=None, vmax=None)
    mngs.plt.configure_mpl(plt, figscale=1.5, fontsize=24)
    fig = plot_topomap(data, CH_MONTAGES_WO_REF, vmin=None, vmax=None, cmap="coolwarm",)

    F_vals = np.random.rand(19, 6)

    plt.show()
