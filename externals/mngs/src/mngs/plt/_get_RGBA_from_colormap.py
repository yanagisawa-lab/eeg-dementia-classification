#!/usr/bin/env python3


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


class ColorGetter:
    # https://stackoverflow.com/questions/26108436/how-can-i-get-the-matplotlib-rgb-color-given-the-colormap-name-boundrynorm-an
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def get_RGBA_from_colormap(val, cmap="Blues", cmap_start_val=0, cmap_stop_val=1):
    ColGetter = ColorGetter(cmap, cmap_start_val, cmap_stop_val)
    return ColGetter.get_rgb(val)
