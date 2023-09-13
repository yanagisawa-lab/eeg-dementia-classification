#!/usr/bin/env python3
import sys

## https://qiita.com/ynakayama/items/c51c2b6d6a3818bdb63f
# def set_trace():
def ipdb():
    from IPython.core.debugger import Pdb

    # Pdb(color_scheme="Linux").set_trace(sys._getframe().f_back)
    Pdb().set_trace(sys._getframe().f_back)


def debug(f, *args, **kwargs):
    from IPython.core.debugger import Pdb

    # pdb = Pdb(color_scheme="Linux")
    pdb = Pdb()
    return pdb.runcall(f, *args, **kwargs)


################################################################################
## Debugging
################################################################################
def paste():
    from IPython import embed

    embed()


def reload(module_or_func):
    import importlib
    import sys

    try:
        importlib.reload(module_orfunc)
    except:
        pass

    try:
        importlib.reload(sys.modules[module_orfunc.__module__])
    except:
        pass


################################################################################
## Debugging
################################################################################
# https://qiita.com/ynakayama/items/c51c2b6d6a3818bdb63f
# def set_trace():
#     from IPython.core.debugger import Pdb

#     Pdb(color_scheme="Linux").set_trace(sys._getframe().f_back)


# def debug(f, *args, **kwargs):
#     from IPython.core.debugger import Pdb

#     pdb = Pdb(color_scheme="Linux")
#     return pdb.runcall(f, *args, **kwargs)


# def ipdb():
#     import ipdb

#     ipdb.set_trace()


# def reload_func(func):
#     import importlib
#     import sys

#     # importlib.reload(sys.modules['foo'])
#     importlib.reload(sys.modules[func.__module__])
#     # from foo import bar
#     # from sys.modules[func.__module__] import func
#     return func


# def reload_module(module):
#     import importlib

#     module = importlib.reload(module)
#     return module


# def modify_func(func):
#     module = sys.modules[func.__module__]


# def modify_func(fun):
#     modulename = fun.__module__
#     module = __import__(modulename)


# def modify_func(fun):
#     import inspect

#     module = inspect.getmodule(fun)
