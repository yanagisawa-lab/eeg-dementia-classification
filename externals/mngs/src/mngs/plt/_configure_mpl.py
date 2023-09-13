#!/usr/bin/env python3


def configure_mpl(
    plt,
    dpi=100,
    figsize=(16.2, 10),
    figscale=1.0,
    fontsize=16,
    labelsize="same",
    legendfontsize="xx-small",
    tick_size="auto",
    tick_width="auto",
    hide_spines=False,
):
    """
    Configures matplotlib for often used format.

    keyword arguments:
        figsize:
            (width, height) [cm]. 18.1 cm () for 2 columns, 8.7 cm for 1 columns look good for papers.
            Default: (16.2, 10) (= the golden ratio in landscape)

        fontsize:
            Default: 20

        labelsize:
            int (Default: 'same' as the fontsize)

        legendfontsize:
            Default: "xx-small"

        tick_size:
            Major tick size [mm]. 0.8 looks good for papers. (Default: 'auto')

        tick_width:
            Major tick width [mm]. 0.2 looks good for papers. (Default: 'auto')

        hide_spines:
            Hides the top and right spines of the axes. (Default: False)



    Example:
        import matplotlib.pyplot as plt
        import numpy as np

        configure_mpl(plt, figsize=(5, 4), fontsize=7, legendfontsize='small')
        plt.plot(np.random.rand(50))
        plt.show()
    """

    ## shrink the figure a little
    shrink_scale = 0.90

    ## Scales figsize
    figsize_cm = (
        figsize[0] * figscale,
        figsize[1] * figscale,
    )

    figsize_inch = (
        figsize_cm[0] / 2.54,
        figsize_cm[1] / 2.54,
    )

    ## Calculates tick size and width from mm unit.
    dots_on_1_inch_line = dpi ** 0.5
    dots_on_1_cm_line = dots_on_1_inch_line / 2.54
    dots_on_1_mm_line = dots_on_1_cm_line / 10.0

    if tick_size != "auto":
        tick_size /= dots_on_1_mm_line
    if tick_width != "auto":
        tick_width /= dots_on_1_mm_line

    """
    ## scientific notation
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(g))

    # plt.ticklabel_format(axis="both", style="plain", scilimits=(-3, 3))


    """

    ## Summarize the props as updater_dict
    updater_dict = {
        "figure.dpi": dpi,
        "savefig.dpi": 300,
        "figure.figsize": (
            figsize_inch[0] * shrink_scale,
            figsize_inch[1] * shrink_scale,
        ),
        "font.size": fontsize,
        "axes.labelsize": fontsize if labelsize == "same" else labelsize,
        "xtick.labelsize": fontsize if labelsize == "same" else labelsize,
        "ytick.labelsize": fontsize if labelsize == "same" else labelsize,
        "axes.titlesize": fontsize if labelsize == "same" else labelsize,
        "legend.fontsize": legendfontsize,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": not hide_spines,
        "axes.spines.right": not hide_spines,
        # "figure.autolayout": True,
    }

    ## Tick
    if tick_size != "auto":
        updater_dict["xtick.major.size"] = tick_size
        updater_dict["ytick.major.size"] = tick_size
    if tick_width != "auto":
        updater_dict["xtick.major.width"] = tick_width
        updater_dict["ytick.major.width"] = tick_width
    """
    ## As the example below, number of ticks can be set for each "ax" object.
    max_n_xticks = 4
    max_n_yticks = 4
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_n_xticks))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(max_n_yticks))
    """

    for k, v in updater_dict.items():
        plt.rcParams[k] = v

    updater_dict["figure.figsize"] = str(figsize_cm) + " [cm]"

    print("\nMatplotilb has been configured as follows:\n{}.\n".format(updater_dict))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    configure_mpl(plt, figsize=(10, 8), fontsize=7, legendfontsize="small")
    plt.plot(np.random.rand(50))
    plt.show()

    # def configure_mpl(
    #     plt, figsize=(1.62, 1), figscale=10, fontsize=20, legendfontsize="xx-small"
    # ):
    #     figbasesize = figsize  # rename for readability

    #     if figsize != (1.62, 1):
    #         figscale = 1

    #     updater_dict = {
    #         # "figure.figsize": (round(1.62 * figscale, 1), round(1 * figscale, 1)),
    #         "figure.figsize": (
    #             round(figsize[0] * figscale, 1),
    #             round(figsize[1] * figscale, 1),
    #         ),
    #         "font.size": fontsize,
    #         "legend.fontsize": legendfontsize,
    #         "pdf.fonttype": 42,
    #         "ps.fonttype": 42,
    #     }
    #     for k, v in updater_dict.items():
    #         plt.rcParams[k] = v
    #     print("\nMatplotilb has been configured as follows:\n{}.\n".format(updater_dict))

    ## EOF
