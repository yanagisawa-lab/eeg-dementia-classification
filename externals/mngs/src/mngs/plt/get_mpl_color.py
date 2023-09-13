#!/usr/bin/env python3

import numpy as np

def _rgb2bgr(rgb_colors):
    return (rgb_colors[1],
            rgb_colors[0],
            rgb_colors[2])

def _cvt_color_name_to_rgb(color_name):
    RGB_PALLETE_DICT = {
        'blue':(0,128,192),
        'red':(255,70,50),
        'pink':(255,150,200),
        'green':(20,180,20),
        'yellow':(230,160,20),
        'glay':(128,128,128),
        'parple':(200,50,255),
        'light_blue':(20,200,200),
        'blown':(128,0,0),
        'navy':(0,0,100),
    }
    return RGB_PALLETE_DICT[color_name]


def _cvt_color_name_to_bgr(color_name):
    return _rgb2bgr(_cvt_color_name_to_rgb(color_name))


def get_mpl_color(color_name):
    color_code = _cvt_color_name_to_rgb(color_name)
    return np.array(color_code) / 255



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    ## Demo plot using get_color
    th = np.linspace(0, 2*np.pi, 128)
    plt.plot(th, np.cos(th), color=get_mpl_color('green')); plt.show()

    ## EOF

