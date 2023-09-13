#!/usr/bin/env python

import mne
import numpy as np
import matplotlib.pyplot as plt


def plot_topomap_bands(
    data,
    montages_wo_ref,
    band_names,
    vmin=None,
    vmax=None,
    unit_label=None,
):
    """
    data: (n_channels, n_bands)
    """
    montages_wo_ref = [m.capitalize() for m in montages_wo_ref]
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()        
    
    n_bands = data.shape[-1]
    TEN_TWENTY = mne.channels.make_standard_montage(
        "standard_1020",
        head_size=0.080,
    )
    CH_POSI_ARR_XYZ = np.array(
        [TEN_TWENTY.get_positions()["ch_pos"][mm] for mm in montages_wo_ref]
    )
    CH_POSI_ARR_XY = CH_POSI_ARR_XYZ[:, :2]

    ## Plots
    fig, axes = plt.subplots(ncols=n_bands)
    for i_ax, ax in enumerate(axes):
        im, cm = mne.viz.plot_topomap(
            data[:, i_ax], CH_POSI_ARR_XY, axes=ax, show=False, vmin=vmin, vmax=vmax
        )
        ax.set_title(band_names[i_ax])
    # manually fiddle the position of colorbar
    ax_x_start = 0.9  # .95
    ax_x_width = 0.04
    ax_y_start = 0.1
    ax_y_height = 0.3  # 0.9
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    fontsize = 12
    clb.ax.set_title(unit_label, fontsize=fontsize)  # title on top of colorbar

    return fig, axes


if __name__ == "__main__":
    MONTAGES = [
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
    MONTAGES_WO_REF = [ch_montage.split("-")[0] for ch_montage in MONTAGES]
    BAND_NAMES = [
        "$\delta$",
        "$\\theta$",
        "$low\ \\alpha$",        
        "$high\ \\alpha$",
        "$\\beta$",
        "$\\gamma$",
    ]

    ## Creates demo data
    data = np.random.rand(len(MONTAGES_WO_REF), len(BAND_NAMES))

    ## Plots
    fig, axes = plot_topomap_bands(
        data, MONTAGES_WO_REF, BAND_NAMES, vmin=None, vmax=None
    )
    fig.show()
