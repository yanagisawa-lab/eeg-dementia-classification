#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def annotated_heatmap(
    cm,
    labels=None,
    title=None,
    cmap="Blues",
    norm=None,
    xlabel=None,
    ylabel=None,
    annot=True,
):
    df = pd.DataFrame(data=cm)

    if labels is not None:
        df.index = labels
        df.columns = labels

    fig, ax = plt.subplots()
    res = sns.heatmap(
        df,
        annot=annot,
        fmt=".3f",
        cmap=cmap,
        norm=norm,
    )  # cbar_kws={"shrink": 0.82}
    res.invert_yaxis()

    # make the frame invisible
    for _, spine in res.spines.items():
        spine.set_visible(False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors

    labels = ["T2T", "F2T", "T2F", "F2F"]
    arr = np.random.randint(0, 10, len(labels) ** 2).reshape(len(labels), len(labels))

    ## quantized, arbitrary range colormap you want
    cmap = colors.ListedColormap(
        ["navy", "royalblue", "lightsteelblue", "beige"],
    )
    norm = colors.BoundaryNorm([2, 4, 6, 8], cmap.N - 1)

    fig = annotated_heatmap(arr, cmap=cmap, norm=norm, labels=labels)
    fig.show()
