#!/usr/bin/env python3

import inspect
import os

import mngs
import warnings


if "general" in __file__:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            '\n"mngs.general.path" will be removed. '
            'Please use "mngs.io.path" instead.',
            PendingDeprecationWarning,
        )


################################################################################
## PATH
################################################################################
def get_this_fpath(when_ipython="/tmp/fake.py"):
    __file__ = inspect.stack()[1].filename
    if "ipython" in __file__:  # for ipython
        __file__ = when_ipython  # "/tmp/fake.py"
    return __file__


def mk_spath(sfname, makedirs=False):

    __file__ = inspect.stack()[1].filename
    if "ipython" in __file__:  # for ipython
        __file__ = f'/tmp/fake-{os.getenv("USER")}.py'

    # __file__ = get_current_file_name()

    ## spath
    fpath = __file__
    fdir, fname, _ = split_fpath(fpath)
    sdir = fdir + fname + "/"
    spath = sdir + sfname

    if makedirs:
        os.makedirs(mngs.general.split_fpath(spath)[0], exist_ok=True)

    return spath


def find_the_git_root_dir():
    import git

    repo = git.Repo(".", search_parent_directories=True)
    return repo.working_tree_dir


def split_fpath(fpath):
    """Split a file path to (1) the directory path, (2) the file name, and (3) the file extention
    Example:
        dirname, fname, ext = split_fpath('../data/01/day1/split_octave/2kHz_mat/tt8-2.mat')
        print(dirname) # '../data/01/day1/split_octave/2kHz_mat/'
        print(fname) # 'tt8-2'
        print(ext) # '.mat'
    """
    dirname = os.path.dirname(fpath) + "/"
    base = os.path.basename(fpath)
    fname, ext = os.path.splitext(base)
    return dirname, fname, ext

def touch(fpath):
    import pathlib
    return pathlib.Path(fpath).touch()
    
