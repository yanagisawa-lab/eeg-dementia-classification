#!/usr/bin/env python3


def ax_extend(ax, x_extend_ratio=1.0, y_extend_ratio=1.0):
    ## Original coordinates
    bbox = ax.get_position()
    left_orig = bbox.x0
    bottom_orig = bbox.y0
    width_orig = bbox.x1 - bbox.x0
    height_orig = bbox.y1 - bbox.y0
    g_orig = (left_orig + width_orig / 2.0, bottom_orig + height_orig / 2.0)

    ## Target coordinates
    g_tgt = g_orig
    width_tgt = width_orig * x_extend_ratio
    height_tgt = height_orig * y_extend_ratio
    left_tgt = g_tgt[0] - width_tgt / 2
    bottom_tgt = g_tgt[1] - height_tgt / 2

    ax.set_position(
        [
            left_tgt,
            bottom_tgt,
            width_tgt,
            height_tgt,
        ]
    )
    return ax


# """
# ====================
# Demo Fixed Size Axes
# ====================
# """

# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import Divider, Size

# ###############################################################################


# fig = plt.figure(figsize=(6, 6))

# # The first items are for padding and the second items are for the axes.
# # sizes are in inch.
# h = [Size.Fixed(1.0), Size.Fixed(4.5)]
# v = [Size.Fixed(0.7), Size.Fixed(5.0)]

# divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
# # The width and height of the rectangle are ignored.

# ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))

# ax.plot([1, 2, 3])
# fig.show()

# ###############################################################################


# fig = plt.figure(figsize=(6, 6))

# # The first & third items are for padding and the second items are for the
# # axes. Sizes are in inches.
# h = [Size.Fixed(1.0), Size.Scaled(1.5), Size.Fixed(0.8)]
# v = [Size.Fixed(0.7), Size.Scaled(1.5), Size.Fixed(0.3)]

# divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
# # The width and height of the rectangle are ignored.

# ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))

# ax.plot([1, 2, 3])

# plt.show()


# def ax_extend(
#     fig,
#     ax,
#     x_extend_ratio=1.00,
#     y_extend_ratio=1.00,
# ):

#     ################################################################################
#     ## delete here
#     import matplotlib.pyplot as plt
#     import numpy as np

#     fig, ax = plt.subplots()
#     x_extend_ratio = 1.00
#     y_extend_ratio = 1.00
#     ################################################################################

#     bbox = ax.get_position()

#     ## Calculates delta ratios
#     fig_width_inch, fig_height_inch = fig.get_size_inches()

#     # x_delta_offset_inch = float(x_delta_offset_cm) / 2.54
#     # y_delta_offset_inch = float(y_delta_offset_cm) / 2.54

#     # x_delta_offset_ratio = x_delta_offset_inch / fig_width_inch
#     # y_delta_offset_ratio = y_delta_offset_inch / fig_width_inch

#     ## Determines updated bbox position
#     left = bbox.x0 + x_delta_offset_ratio
#     bottom = bbox.y0 + y_delta_offset_ratio
#     width = bbox.x1 - bbox.x0
#     height = bbox.y1 - bbox.y0

#     if dragh:
#         width -= x_delta_offset_ratio

#     if dragv:
#         height -= y_delta_offset_ratio

#     ax.set_position(
#         [
#             left,
#             bottom,
#             width,
#             height,
#         ]
#     )

#     return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2)
    ax = axes[1]
    ax = ax_extend(ax, 0.75, 1.1)
    fig.show()
