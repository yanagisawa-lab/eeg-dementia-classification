#!/usr/bin/env python3


import matplotlib


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    # https://stackoverflow.com/questions/42656139/set-scientific-notation-with-fixed-exponent-and-significant-digits-for-multiple
    # def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
    def __init__(self, order=0, fformat="%1.0d", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText
        )

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format


def ax_scientific_notation(
    ax,
    order,
    fformat="%1.0d",
    x=False,
    y=False,
    scilimits=(-3, 3),
):
    """
    Change the expression of the x- or y-axis to the scientific notation like *10^3
    , where 3 is the first argument, order.

    Example:
        order = 4 # 10^4
        ax = ax_scientific_notation(
                 ax,
                 order,
                 fformat="%1.0d",
                 x=True,
                 y=False,
                 scilimits=(-3, 3),
    """

    if x == True:
        ax.xaxis.set_major_formatter(OOMFormatter(order=order, fformat=fformat))
        ax.ticklabel_format(axis="x", style="sci", scilimits=scilimits)
    if y == True:
        ax.yaxis.set_major_formatter(OOMFormatter(order=order, fformat=fformat))
        ax.ticklabel_format(axis="y", style="sci", scilimits=scilimits)

    return ax
