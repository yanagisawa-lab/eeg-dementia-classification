#!/usr/bin/env python3


def to_the_latex_style(str_or_num):
    """
    Example:
        print(to_the_latex_style('aaa'))
        # '$aaa$'
    """
    string = str(str_or_num)
    if (string[0] == "$") and (string[-1] == "$"):
        return string
    else:
        return "${}$".format(string)


def add_hat_in_the_latex_style(str_or_num):
    """
    Example:
        print(add_hat_in_the_latex_style('aaa'))
        # '$\\hat{aaa}$'
    """
    return to_the_latex_style("\hat{%s}" % str_or_num)
