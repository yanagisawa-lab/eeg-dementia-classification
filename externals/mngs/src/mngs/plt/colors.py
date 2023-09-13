#!/usr/bin/env python3

import numpy as np


def RGB2RGBA(RGB, alpha=0, round=3):
    RGB = np.array(RGB).astype(float)
    RGB /= 255
    return (*RGB.round(round), alpha)


RGB_d = {
    "blue": [0, 128, 192],
    "red": [255, 70, 50],
    "pink": [255, 150, 200],
    "green": [20, 180, 20],
    "yellow": [230, 160, 20],
    "gray": [128, 128, 128],
    "purple": [200, 50, 255],
    "light_blue": [20, 200, 200],
    "blown": [128, 0, 0],
    "navy": [0, 0, 100],
}


def cycle_color(i_color):
    COLORS_10_STR = list(RGB_d.keys())
    n_colors = len(COLORS_10_STR)
    return COLORS_10_STR[i_color % n_colors]


def to_RGB(c):
    RGB_d = {
        "blue": [0, 128, 192],
        "red": [255, 70, 50],
        "pink": [255, 150, 200],
        "green": [20, 180, 20],
        "yellow": [230, 160, 20],
        "gray": [128, 128, 128],
        "purple": [200, 50, 255],
        "light_blue": [20, 200, 200],
        "blown": [128, 0, 0],
        "navy": [0, 0, 100],
    }
    return RGB_d[c]


def to_RGBA(c, alpha=0.5):
    RGBA_d = {
        "blue": (0.0, 0.502, 0.753, alpha),
        "red": (1.0, 0.275, 0.196, alpha),
        "pink": (1.0, 0.588, 0.784, alpha),
        "green": (0.078, 0.706, 0.078, alpha),
        "yellow": (0.902, 0.627, 0.078, alpha),
        "gray": (0.502, 0.502, 0.502, alpha),
        "purple": (0.784, 0.196, 1.0, alpha),
        "light_blue": (0.078, 0.784, 0.784, alpha),
        "blown": (0.502, 0.0, 0.0, alpha),
        "navy": (0.0, 0.0, 0.392, alpha),
    }
    return RGBA_d[c]
