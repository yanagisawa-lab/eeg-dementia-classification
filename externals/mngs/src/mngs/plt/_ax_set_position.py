#!/usr/bin/env python3


def ax_set_position(
    fig,
    ax,
    x_delta_offset_cm,
    y_delta_offset_cm,
    dragh=False,
    dragv=False,
):

    bbox = ax.get_position()

    ## Calculates delta ratios
    fig_width_inch, fig_height_inch = fig.get_size_inches()

    x_delta_offset_inch = float(x_delta_offset_cm) / 2.54
    y_delta_offset_inch = float(y_delta_offset_cm) / 2.54

    x_delta_offset_ratio = x_delta_offset_inch / fig_width_inch
    y_delta_offset_ratio = y_delta_offset_inch / fig_width_inch

    ## Determines updated bbox position
    left = bbox.x0 + x_delta_offset_ratio
    bottom = bbox.y0 + y_delta_offset_ratio
    width = bbox.x1 - bbox.x0
    height = bbox.y1 - bbox.y0

    if dragh:
        width -= x_delta_offset_ratio

    if dragv:
        height -= y_delta_offset_ratio

    ax.set_position(
        [
            left,
            bottom,
            width,
            height,
        ]
    )

    return ax
