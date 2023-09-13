#!/usr/bin/env python3

import mngs


################################################################################
## Reproducibility
################################################################################
def fix_seeds(os=None, random=None, np=None, torch=None, tf=None, seed=42, show=True):
    # https://github.com/lucidrains/vit-pytorch/blob/main/examples/cats_and_dogs.ipynb
    if os is not None:
        import os

        os.environ["PYTHONHASHSEED"] = str(seed)

    if random is not None:
        random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)

    if tf is not None:
        tf.random.set_seed(seed)

    if show:
        print("\nRandom seeds have been fixed as {}\n".format(seed))


def gen_ID(N=8):
    import random
    import string
    from datetime import datetime

    now = datetime.now()
    # now_str = now.strftime('%Y-%m-%d-%H:%M')
    today_str = now.strftime("%Y-%m%d")
    randlst = [random.choice(string.ascii_letters + string.digits) for i in range(N)]
    rand_str = "".join(randlst)
    return today_str + "_" + rand_str
    return rand_str


def gen_timestamp():
    from datetime import datetime

    now = datetime.now()
    now_str = now.strftime("%Y-%m%d-%H%M")
    return now_str


class Tee(object):
    """Example:
    import sys

    sys.stdout = Tee(sys.stdout, "stdout.txt")
    sys.stderr = Tee(sys.stderr, "stderr.txt")

    print("abc") # stdout
    print(1 / 0) # stderr
    # cat stdout.txt
    # cat stderr.txt
    """

    def __init__(self, sys_stdout_or_stderr, spath):
        self._files = [sys_stdout_or_stderr, open(spath, "w")]

    def __getattr__(self, attr, *args):
        return self._wrap(attr, *args)

    def _wrap(self, attr, *args):
        def g(*a, **kw):
            for f in self._files:
                res = getattr(f, attr, *args)(*a, **kw)
            return res

        return g


def tee(sys, sdir=None):
    """
    import sys

    sys.stdout, sys.stderr = tee(sys)

    print("abc")  # stdout
    print(1 / 0)  # stderr
    """

    import inspect
    import os

    ####################
    ## Determines sdir
    ####################
    if sdir is None:
        __file__ = inspect.stack()[1].filename
        if "ipython" in __file__:
            __file__ = "/tmp/fake.py"
        spath = __file__

        _sdir, sfname, _ = mngs.general.split_fpath(spath)
        sdir = _sdir + sfname + "/log/"

    os.makedirs(sdir, exist_ok=True)

    spath_stdout = sdir + "stdout.log"
    spath_stderr = sdir + "stderr.log"
    os.makedirs(mngs.general.split_fpath(spath_stdout)[0], exist_ok=True)

    sys_stdout = Tee(sys.stdout, spath_stdout)
    sys_stderr = Tee(sys.stdout, spath_stderr)

    print(
        "\nStandard Output/Error are going to be logged in the followings: \n  - {}\n  - {}\n".format(
            spath_stdout, spath_stderr
        )
    )
    return sys_stdout, sys_stderr
