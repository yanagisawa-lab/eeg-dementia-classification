#!/usr/bin/env python3

import collections
import math
import os
import re
import shutil
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict

import mngs
import numpy as np
import pandas as pd
import torch
from natsort import natsorted


################################################################################
## strings
################################################################################
def decapitalize(s):
    if not s:  # check that s is not empty string
        return s
    return s[0].lower() + s[1:]


def connect_strs(strs, filler="_"):  # connect_nums also works as connect_strs
    """
    Example:
        print(connect_strs(['a', 'b', 'c'], filler='_'))
        # 'a_b_c'
    """
    if isinstance(strs, list) or isinstance(strs, tuple):
        connected = ""
        for s in strs:
            connected += filler + s
        return connected[len(filler) :]
    else:
        return strs


def connect_nums(nums, filler="_"):
    """
    Example:
        print(connect_nums([1, 2, 3], filler='_'))
        # '1_2_3'
    """
    if isinstance(nums, list) or isinstance(nums, tuple):
        connected = ""
        for n in nums:
            connected += filler + str(n)
        return connected[len(filler) :]
    else:
        return nums


def squeeze_spaces(string, pattern=" +", repl=" "):
    """Return the string obtained by replacing the leftmost
    non-overlapping occurrences of the pattern in string by the
    replacement repl.  repl can be either a string or a callable;
    if a string, backslash escapes in it are processed.  If it is
    a callable, it's passed the Match object and must return
    a replacement string to be used.
    """
    # return re.sub(" +", " ", string)
    return re.sub(pattern, repl, string)


def search(patterns, strings, only_perfect_match=False):
    """
    regular expression is acceptable for patterns.

    Example:
        patterns = ['orange', 'banana']
        strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
        print(search(patterns, strings))
        # ([1, 4, 5], ['orange', 'banana', 'orange_juice'])

        patterns = 'orange'
        strings = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
        print(search(patterns, strings))
        # ([1, 5], ['orange', 'orange_juice'])
    """

    ## For single string objects
    def to_list(s_or_p):
        if isinstance(s_or_p, collections.abc.KeysView):
            s_or_p = list(s_or_p)

        elif not isinstance(s_or_p, (list, tuple, pd.core.indexes.base.Index)):
            s_or_p = [s_or_p]

        return s_or_p

    patterns = to_list(patterns)
    strings = to_list(strings)

    ## Main
    if not only_perfect_match:
        indi_matched = []
        for pattern in patterns:
            for i_str, string in enumerate(strings):
                m = re.search(pattern, string)
                if m is not None:
                    indi_matched.append(i_str)
    else:
        indi_matched = []
        for pattern in patterns:
            for i_str, string in enumerate(strings):
                if pattern == string:
                    indi_matched.append(i_str)

    ## Sorts the indices according to the original strings
    indi_matched = natsorted(indi_matched)
    keys_matched = list(np.array(strings)[indi_matched])
    return indi_matched, keys_matched


def grep(str_list, search_key):
    """
    Example:
        str_list = ['apple', 'orange', 'apple', 'apple_juice', 'banana', 'orange_juice']
        search_key = 'orange'
        print(grep(str_list, search_key))
        # ([1, 5], ['orange', 'orange_juice'])
    """
    matched_keys = []
    indi = []
    for ii, string in enumerate(str_list):
        m = re.search(search_key, string)
        if m is not None:
            matched_keys.append(string)
            indi.append(ii)
    return indi, matched_keys


def pop_keys(keys_list, keys_to_pop):
    """
    keys_list = ['a', 'b', 'c', 'd', 'e', 'bde']
    keys_to_pop = ['b', 'd']
    pop_keys(keys_list, keys_to_pop)
    """
    indi_to_remain = [k not in keys_to_pop for k in keys_list]
    keys_remainded_list = list(np.array(keys_list)[list(indi_to_remain)])
    return keys_remainded_list


def fmt_size(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


################################################################################
## list
################################################################################
def take_the_closest(list_obj, num_insert):
    """
    Assumes list_obj is sorted. Returns the closest value to num.
    If the same number is included in list_obj, the smaller number is returned.

    Example:
        list_obj = np.array([0, 1, 1, 2, 3, 3])
        num = 1.2
        closest_num, closest_pos = take_the_closest(list_obj, num)
        print(closest_num, closest_pos)
        # 1 2

        list_obj = np.array([0, 1, 1, 2, 3, 3])
        num = 1
        closest_num, closest_pos = take_the_closest(list_obj, num)
        print(closest_num, closest_pos)
        # 1 1
    """
    if math.isnan(num_insert):
        closest_num = np.nan
        closest_pos = np.nan

    pos_num_insert = bisect_left(list_obj, num_insert)

    if pos_num_insert == 0:  # When the insertion is at the first position
        closest_num = list_obj[0]
        closest_pos = pos_num_insert  # 0

    if pos_num_insert == len(list_obj):  # When the insertion is at the last position
        closest_num = list_obj[-1]
        closest_pos = pos_num_insert  # len(list_obj)

    else:  # When the insertion is anywhere between the first and the last positions
        pos_before = pos_num_insert - 1

        before_num = list_obj[pos_before]
        after_num = list_obj[pos_num_insert]

        delta_after = abs(after_num - num_insert)
        delta_before = abs(before_num - num_insert)

        if np.abs(delta_after) < np.abs(delta_before):
            closest_num = after_num
            closest_pos = pos_num_insert

        else:
            closest_num = before_num
            closest_pos = pos_before

    return closest_num, closest_pos


################################################################################
## mutable
################################################################################
def isclose(mutable_a, mutable_b):
    return [math.isclose(a, b) for a, b in zip(mutable_a, mutable_b)]


################################################################################
## dictionary
################################################################################
def merge_dicts_wo_overlaps(*dicts):
    merged_dict = {} # init
    for dict in dicts:
        assert mngs.general.search(merged_dict.keys(), dict.keys(), only_perfect_match=True) == ([], [])
        merged_dict.update(dict)
    return merged_dict


def listed_dict(keys=None):  # Is there a better name?
    """
    Example 1:
        import random
        random.seed(42)
        d = listed_dict()
        for _ in range(10):
            d['a'].append(random.randint(0, 10))
        print(d)
        # defaultdict(<class 'list'>, {'a': [10, 1, 0, 4, 3, 3, 2, 1, 10, 8]})

    Example 2:
        import random
        random.seed(42)
        keys = ['a', 'b', 'c']
        d = listed_dict(keys)
        for _ in range(10):
            d['a'].append(random.randint(0, 10))
            d['b'].append(random.randint(0, 10))
            d['c'].append(random.randint(0, 10))
        print(d)
        # defaultdict(<class 'list'>, {'a': [10, 4, 2, 8, 6, 1, 8, 8, 8, 7],
        #                              'b': [1, 3, 1, 1, 0, 3, 9, 3, 6, 9],
        #                              'c': [0, 3, 10, 9, 0, 3, 0, 10, 3, 4]})
    """
    dict_list = defaultdict(list)
    # initialize with keys if possible
    if keys is not None:
        for k in keys:
            dict_list[k] = []
    return dict_list


################################################################################
## variables
################################################################################
def is_defined_global(x_str):
    """
    Example:
        print(is_defined('a'))
        # False

        a = 5
        print(is_defined('a'))
        # True
    """
    return x_str in globals()


def is_defined_local(x_str):
    """
    Example:
        print(is_defined('a'))
        # False

        a = 5
        print(is_defined('a'))
        # True
    """
    return x_str in locals()


# def does_exist(suspicious_var_str):
#     return suspicious_var_str in globals()


################################################################################
## versioning
################################################################################
def is_later_or_equal(package, tgt_version, format="MAJOR.MINOR.PATCH"):
    import mngs
    import numpy as np

    indi, matched = mngs.general.search(["MAJOR", "MINOR", "PATCH"], format.split("."))
    imp_major, imp_minor, imp_patch = [
        int(v) for v in np.array(package.__version__.split("."))[indi]
    ]
    tgt_major, tgt_minor, tgt_patch = [
        int(v) for v in np.array(tgt_version.split("."))[indi]
    ]

    print(
        f"\npackage: {package.__name__}\n"
        f"target_version: {tgt_version}\n"
        f"imported_version: {imp_major}.{imp_minor}.{imp_patch}\n"
    )

    ## Mjorr
    if imp_major > tgt_major:
        return True

    if imp_major < tgt_major:
        return False

    if imp_major == tgt_major:

        ## Minor
        if imp_minor > tgt_minor:
            return True

        if imp_minor < tgt_minor:
            return False

        if imp_minor == tgt_minor:

            ## Patch
            if imp_patch > tgt_patch:
                return True
            if imp_patch < tgt_patch:
                return False
            if imp_patch == tgt_patch:
                return True


################################################################################
## File
################################################################################
def _copy_a_file(src, dst, allow_overwrite=False):
    if src == "/dev/null":
        print(f"\n/dev/null was not copied.\n")

    else:

        if dst.endswith("/"):
            _, src_fname, src_ext = mngs.general.split_fpath(src)
            # src_fname = src + src_ext
            dst = dst + src_fname + src_ext

        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            print(f'\nCopied "{src}" to "{dst}".\n')

        else:
            if allow_overwrite:
                shutil.copyfile(src, dst)
                print(f'\nCopied "{src}" to "{dst}" (overwritten).\n')

            if not allow_overwrite:
                print(f'\n"{dst}" exists and copying from "{src}" was aborted.\n')


def copy_files(src_files, dists, allow_overwrite=False):
    if isinstance(src_files, str):
        src_files = [src_files]

    if isinstance(dists, str):
        dists = [dists]

    for sf in src_files:
        for dst in dists:
            _copy_a_file(sf, dst, allow_overwrite=allow_overwrite)


def copy_the_file(sdir):  # dst
    __file__ = inspect.stack()[1].filename
    _, fname, ext = mngs.general.split_fpath(__file__)

    dst = sdir + fname + ext

    if "ipython" not in __file__:  # for ipython
        # shutil.copyfile(__file__, dst)
        # print(f"Saved to: {dst}")
        _copy_a_file(__file__, dst)

def is_nan(X):
    if isinstance(X, pd.DataFrame):
        if X.isna().any().any():
            raise ValueError("NaN was found in X")
    elif isinstance(X, np.ndarray):
        if np.isnan(X).any():        
            raise ValueError("NaN was found in X")
    elif torch.is_tensor(X):
        if X.isnan().any():        
            raise ValueError("NaN was found in X")
    elif isinstance(X, (float, int)):
        if math.isnan(X):
            raise ValueError("X was NaN")

